"""Independent diagnostic utilities; portable actual-gate simulator and independently specified classical recurrence.

Sparse simulation keeps complex amplitudes and removes exactly-zero entries only.
No MPS, amplitude cutoff, classical trajectory sampling, or production edits.
"""
import sys, math, time, json, hashlib, os
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, replace
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from config import default_config
from unitary_circuit import build_unitary_circuit_qiskit

DIRS=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

def reference(cfg,steps,retain_joint=False):
    """Direct closed-interval finite-step specification, not production helpers.

    Extends prior independent exact_classical by retaining terminal joint states,
    first termination times and implementing the declared downscatter Bernoulli.
    """
    g=cfg.geom; s=cfg.source
    live={(s.x0,s.y0,s.z0,s.dir0,s.g0):1.0}
    terminal=defaultdict(float); rows=[]
    for t in range(1,steps+1):
        nxt=defaultdict(float); first=[0.0]*4
        for (x,y,z,d,e),p in live.items():
            if x in (0,g.nx-1) or y in (0,g.ny-1) or z in (0,g.nz-1):
                terminal[(3,x,y,z,d,e)]+=p;first[3]+=p;continue
            if x==g.detector_x and g.roi_y0<=y<=g.roi_y1 and g.roi_z0<=z<=g.roi_z1:
                terminal[(1,x,y,z,d,e)]+=p;first[1]+=p;continue
            name='concrete'
            if g.slab_x0<=x<=g.slab_x1:
                name='air' if g.duct_y0<=y<=g.duct_y1 and g.duct_z0<=z<=g.duct_z1 else 'tungsten'
            m=getattr(cfg.mats,name);pi=m.p_int[e];pa=m.p_abs[e]
            mass=p*pi*pa
            terminal[(2,x,y,z,d,e)]+=mass;first[2]+=mass
            dx,dy,dz=DIRS[d]
            nxt[(x+dx,y+dy,z+dz,d,e)]+=p*(1-pi)
            q=m.q_down[e] if e+1<cfg.groups.n_groups else 0.0
            for dn,(dx,dy,dz) in enumerate(DIRS):
                mass=p*pi*(1-pa)/6
                nxt[(x+dx,y+dy,z+dz,dn,e)]+=mass*(1-q)
                if q:nxt[(x+dx,y+dy,z+dz,dn,e+1)]+=mass*q
        live={k:v for k,v in nxt.items() if v!=0}
        probs=[math.fsum(live.values()),0.0,0.0,0.0]
        for key,p in terminal.items():probs[key[0]]+=p
        rows.append({'steps':t,'probabilities':probs,'first_termination':first,
                     'norm_error':abs(math.fsum(probs)-1),'live_states':len(live)})
    joint=dict(terminal)
    joint.update({(0,)+k:p for k,p in live.items()})
    result={'order':['survive','detect','absorb','boundary'],'config':asdict(cfg),'rows':rows}
    if retain_joint:result['joint']=joint
    return result

PRIMITIVES={'x','y','z','h','s','sdg','t','tdg','p','rx','ry','rz','u','unitary','id','swap'}

def flattened(qc,mapping=None):
    """Yield actual gate matrices/controls; fail on unsupported nonunitary ops."""
    mapping=list(range(qc.num_qubits)) if mapping is None else mapping
    phase=float(qc.global_phase)
    if phase:yield ('phase',np.exp(1j*phase))
    for ins in qc.data:
        op=ins.operation;bits=[mapping[qc.find_bit(q).index] for q in ins.qubits]
        if op.name=='barrier':continue
        if op.name in ('measure','reset','initialize'):raise ValueError('unsupported nonunitary '+op.name)
        nc=getattr(op,'num_ctrl_qubits',0)
        base=getattr(op,'base_gate',op)
        if base.name in PRIMITIVES:
            cs=getattr(op,'ctrl_state',(1<<nc)-1)
            if cs is None:cs=(1<<nc)-1
            controls=bits[:nc];targets=bits[nc:]
            if len(targets)>3:raise ValueError('matrix too large '+base.name)
            cm=sum(1<<b for b in controls)
            cv=sum(((int(cs)>>i)&1)<<b for i,b in enumerate(controls))
            yield ('gate',base.name,cm,cv,targets,np.asarray(base.to_matrix(),complex))
        elif op.definition is not None:
            yield from flattened(op.definition,bits)
        else:raise ValueError('unsupported gate '+op.name)

def apply_instruction(state, item):
    if item[0]=='phase':return {k:a*item[1] for k,a in state.items()}
    _,name,cm,cv,targets,mat=item
    if name=='x':
        mask=1<<targets[0]
        return {(k^mask if (k&cm)==cv else k):a for k,a in state.items()}
    if np.count_nonzero(mat-np.diag(np.diag(mat)))==0:
        return {k:a*mat[sum(((k>>b)&1)<<i for i,b in enumerate(targets)),sum(((k>>b)&1)<<i for i,b in enumerate(targets))] if (k&cm)==cv else a for k,a in state.items()}
    tm=sum(1<<b for b in targets)
    shifts=[sum(((i>>j)&1)<<b for j,b in enumerate(targets)) for i in range(1<<len(targets))]
    columns=[[(shifts[r],complex(c)) for r,c in enumerate(mat[:,col]) if c!=0] for col in range(len(shifts))]
    nxt=defaultdict(complex)
    for k,a in state.items():
        if (k&cm)!=cv:nxt[k]+=a;continue
        col=sum(((k>>b)&1)<<i for i,b in enumerate(targets));clear=k&~tm
        for shift,c in columns[col]:nxt[clear|shift]+=c*a
    return {k:a for k,a in nxt.items() if a!=0}

def evolve(qc,state=None,timeout=None,max_states=600000,checkpoints=None,observer=None):
    """Evolve actual gates; timeout=None removes only this call's wall limit.

    An explicitly inherited monotonic deadline remains binding. Normal calls do not create a wall-clock deadline. State-size and all numerical checks remain independent.
    """
    if timeout is not None:
        timeout=float(timeout)
        if not math.isfinite(timeout) or timeout<=0:
            raise ValueError('timeout must be None or a positive finite number')
    parent_text=os.environ.get('VRQO_DEADLINE_MONOTONIC')
    parent_deadline=None if parent_text is None else float(parent_text)
    if parent_deadline is not None and not math.isfinite(parent_deadline):
        raise ValueError('VRQO_DEADLINE_MONOTONIC must be finite when set')
    state={0:1+0j} if state is None else state.copy()
    start=time.monotonic();peak=len(state);ngates=0;observations=[]
    deadline=None if timeout is None else start+timeout
    if parent_deadline is not None:deadline=parent_deadline if deadline is None else min(deadline,parent_deadline)
    def check_limits():
        if len(state)>max_states:raise MemoryError(f'sparse state budget exceeded: {len(state)} > {max_states}')
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError(f'diagnostic timeout: effective deadline reached (timeout_seconds={timeout}, parent_deadline_monotonic={parent_deadline}), peak={peak}, gates={ngates}')
    check_limits()
    if float(qc.global_phase):state={k:a*np.exp(1j*float(qc.global_phase)) for k,a in state.items()}
    wanted=defaultdict(list)
    for point in checkpoints or []:wanted[point['gate_index']].append(point)
    def check(i):
        for point in wanted.get(i,[]):observations.append(dict(point,observables=observer(qc,state,point)))
        check_limits()
    check(0)
    from qiskit import QuantumCircuit
    for i,ins in enumerate(qc.data):
        if ins.clbits or ins.operation.name in ('measure','reset','initialize'):
            raise ValueError('unsupported nonunitary '+ins.operation.name)
        one=QuantumCircuit(qc.num_qubits)
        one.append(ins.operation,[qc.find_bit(q).index for q in ins.qubits])
        for item in flattened(one):
            check_limits()
            state=apply_instruction(state,item);ngates+=1;peak=max(peak,len(state))
            check_limits()
        check(i+1)
    return state,{'seconds':time.monotonic()-start,'peak_basis':peak,'final_basis':len(state),'primitive_gates':ngates,'checkpoints':observations,
        'timeout_seconds':timeout,'parent_deadline_monotonic':parent_deadline,'effective_deadline_monotonic':deadline,'max_states':max_states}

def registers(qc):return {r.name:[qc.find_bit(q).index for q in r] for r in qc.qregs}
def getval(k,bits):return sum(((k>>b)&1)<<i for i,b in enumerate(bits))
def marginal(qc,state,names):
    regs=registers(qc);out=defaultdict(float)
    for k,a in state.items():out[tuple(getval(k,regs[n]) for n in names)]+=abs(a)**2
    return dict(out)

def observe(qc,state,point=None):
    regs=registers(qc);p=[0.0]*4;work=0.0;badwork=0.0;w14=0.0;w16=0.0;joint=defaultdict(float)
    t=(point or {}).get('step',0)
    for k,a in state.items():
        mass=abs(a)**2;s=getval(k,regs['status']);w=getval(k,regs['work']);p[s]+=mass
        work+=mass*(w!=0);badwork+=mass*((w!=0) and s!=1)
        w14+=mass*((w>>14)&1);w16+=mass*((w>>16)&1)
        if point:
            key=(s,getval(k,regs['x']),getval(k,regs['y']),getval(k,regs['z']),getval(k,regs['dir']), (w>>14)&1,(w>>15)&1,(w>>16)&1, (getval(k,regs['tD'])>>t)&1,(getval(k,regs['tA'])>>t)&1)
            joint[key]+=mass
    ans={'probabilities':p,'norm':math.fsum(p),'work_nonzero':work,'work_nonzero_outside_detect':badwork,'w_dir_pred_nonzero':w14,'w_abs_term_nonzero':w16}
    if point:ans['joint_columns']=['status','x','y','z','dir','work14','work15','work16','tD_current','tA_current'];ans['joint']=[list(k)+[v] for k,v in sorted(joint.items())]
    return ans

def build_traced(cfg,steps,profile):
    points=[]
    def trace(frame,event,arg):
        if Path(frame.f_code.co_filename)==SRC/'unitary_circuit.py' and frame.f_code.co_name=='compile_step_kernel':
            if event=='return':points.append({'step':frame.f_locals['t'],'stage':'step_end','source_line':frame.f_lineno,'gate_index':len(frame.f_locals['qc'].data)})
        return trace
    sys.settrace(trace)
    try:qc,info=build_unitary_circuit_qiskit(cfg,steps,kernel_profile=profile)
    finally:sys.settrace(None)
    return qc,info,points

def config_with_probs(cfg,pi,pa,qd=0):
    mats=replace(cfg.mats,**{n:replace(getattr(cfg.mats,n),p_int=(pi,)*cfg.groups.n_groups,p_abs=(pa,)*cfg.groups.n_groups,q_down=(qd,)*cfg.groups.n_groups) for n in ('concrete','tungsten','air')})
    return replace(cfg,mats=mats)
