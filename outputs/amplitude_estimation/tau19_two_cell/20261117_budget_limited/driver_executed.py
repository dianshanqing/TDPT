"""One bounded natural-rare IQAE supplement, reusing frozen actual-gate code.

No production edits. --prepare fixes the protocol before counts; --run uses the
existing Windows Job supervisor. Cached prefixes contain actual complex states,
not analytic amplitudes. Hardware query accounting never discounts this cache.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'VRQO_v2.0.0'
sys.path.insert(0,str(ROOT))
from reference_engine import flattened,apply_instruction,registers
from runtime_identity import source_manifest,software_versions,circuit_fingerprint,normalize
from supervised import _display_record

PROTOCOL=dict(model='q_tau19_two_cell',steps=3,profile='baseline',objective='detect',
              seed=2026111001,relative_error=.01,confidence=.95,shots_per_round=512,
              max_shots=20000,max_queries=10000000,max_power=256,max_states=60000,
              max_actual_primitive_gate_applications=2000000,memory_mib=1536,
              threads=1,norm_work_tolerance=2e-11,wall_deadline_seconds=None)
DEFAULT_OUT=ROOT/'outputs/local_supplement_20261117/iqae_two_cell'


class BudgetExceeded(RuntimeError):
    status='budget_limited'


class ActualGates:
    def __init__(self,Aops,Qops,objective,work_bits,*,max_states,max_gates):
        self.Aops=Aops;self.Qops=Qops;self.objective=objective
        self.work_mask=sum(1<<b for b in work_bits)
        self.max_states=max_states;self.max_gates=max_gates
        self.used=0;self.peak=1;self.cache={};self.evolutions=[]
        self.highest=-1;self.prefix=None;self.active=None

    def evolve(self,ops,state):
        if self.used+len(ops)>self.max_gates:raise BudgetExceeded('actual primitive gate budget')
        for op in ops:
            state=apply_instruction(state,op);self.used+=1;self.peak=max(self.peak,len(state))
            if len(state)>self.max_states:raise BudgetExceeded('complex basis-state budget')
        return state

    def probability(self,k):
        if k in self.cache:return self.cache[k]
        start=time.perf_counter();before=self.used
        reuse=self.highest>=0 and k>self.highest
        needed=(k-self.highest)*len(self.Qops) if reuse else len(self.Aops)+k*len(self.Qops)
        self.active={'requested_power':k,'prefix_power':self.highest if reuse else None,'required_additional_gates':needed}
        if self.used+needed>self.max_gates:raise BudgetExceeded('actual gate budget before requested power')
        if reuse:state=self.prefix;first=self.highest
        else:state=self.evolve(self.Aops,{0:1+0j});first=0
        for unused in range(first,k):state=self.evolve(self.Qops,state)
        norm=math.fsum(abs(a)**2 for a in state.values())
        probability=math.fsum(abs(a)**2 for key,a in state.items() if (key>>self.objective)&1)
        work=math.fsum(abs(a)**2 for key,a in state.items() if key&self.work_mask)
        if not math.isfinite(probability) or not 0<=probability<=1 or abs(norm-1)>2e-11 or work>2e-11:
            raise ValueError(f'invalid unnormalized actual state: p={probability}, norm={norm}, work={work}')
        if k>self.highest:self.highest=k;self.prefix=state
        self.cache[k]=probability
        self.evolutions.append({'power':k,'actual_gate_probability':probability,'raw_norm':norm,
            'norm_residual':abs(norm-1),'work_nonzero_probability':work,'seconds':time.perf_counter()-start,
            'primitive_gate_applications':self.used-before,'peak_basis_so_far':self.peak,
            'final_basis':len(state),'reused_prefix_power':first if reuse else None})
        self.active=None
        return probability


def write_new(path,value):
    with path.open('x',encoding='utf8') as stream:
        json.dump(normalize(value),stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def prepared(out):
    from models import load_model
    out.mkdir(parents=True,exist_ok=False)
    document={'protocol':PROTOCOL,'config':asdict(load_model(PROTOCOL['model'])),
              'production_sources':source_manifest(),'versions':software_versions(),
              'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_used_for_power_or_stop':False,'preselected_before_counts':True,
              'cache_policy':'Reuse an actual evolved prefix only; fresh binomial draws for every batch; no pruning or normalization.',
              'scientific_scope':'Single natural low-probability short transport case; conditional statistical precision, no full numerical certification.'}
    write_new(out/'protocol.json',document)
    print('Protocol fixed; no quantum evolution or random counts executed.',flush=True)


def worker(out):
    import numpy as np
    from models import load_model
    from numerical_runtime import configure_blas_threads
    from unitary_circuit import build_unitary_circuit_qiskit
    from iqae import OracleSpec,build_grover_operator_for_spec,estimate_relative_iqae
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf8'))
    if protocol['protocol']!=PROTOCOL or protocol['production_sources']!=source_manifest():raise ValueError('Protocol/source identity changed')
    if protocol['driver_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Driver changed after preparation')
    configure_blas_threads(1);begin=time.perf_counter();cpu=time.process_time()
    cfg=load_model(PROTOCOL['model']);A,_=build_unitary_circuit_qiskit(cfg,3,kernel_profile='baseline')
    regs=registers(A);Q,obj,_=build_grover_operator_for_spec(A_circuit=A,status_qubits=regs['status'],g_qubits=regs['g'],spec=OracleSpec(good='detect'))
    marked=Q.state_preparation;inverse=A.inverse()
    engine=ActualGates(list(flattened(marked)),list(flattened(Q)),obj[0],regs['work'],
                      max_states=PROTOCOL['max_states'],max_gates=PROTOCOL['max_actual_primitive_gate_applications'])
    batches=[]
    base={'protocol':protocol,'fingerprints':{'A':circuit_fingerprint(A),'A_inverse':circuit_fingerprint(inverse),
          'marked_A':circuit_fingerprint(marked),'Q':circuit_fingerprint(Q)},'qubits':marked.num_qubits,
          'primitive_components':{'marked_A':len(engine.Aops),'Q':len(engine.Qops)},
          'primitive_unit':'Controlled primitive operations from reference_engine.flattened; not hardware two-qubit gates.',
          'full_pipeline_certified':False}
    write_new(out/'circuit_identity.json',base)
    def progress(status,**extra):
        _display_record(out/'progress.json',{'status':status,'measurement_batches':batches,
            'actual_evolutions':engine.evolutions,'active_power':engine.active,
            'actual_primitive_gate_applications':engine.used,'peak_basis':engine.peak,**extra})
    try:
        # Actual A followed by its inverse, before measurement counts.
        state=engine.evolve(list(flattened(A)),{0:1+0j})
        state=engine.evolve(list(flattened(inverse)),state)
        inverse_error=abs(state.get(0,0)-1)**2+math.fsum(abs(a)**2 for key,a in state.items() if key)
        if inverse_error>2e-11:raise ValueError('A inverse A check failed')
        del state
        rng=np.random.default_rng(PROTOCOL['seed'])
        def sample(k,n):
            progress('evolving',requested_power=k)
            try:probability=engine.probability(k)
            except BaseException as exc:progress('evolution_failed',error=repr(exc));raise
            hits=int(rng.binomial(n,probability))
            batches.append({'power':k,'shots':n,'success_count':hits,'actual_gate_probability':probability})
            progress('batch_returned')
            print(f'[ROUND] {len(batches)} power={k} shots={n} hits={hits} actual_gates={engine.used}',flush=True)
            return hits
        result=estimate_relative_iqae(sample,relative_error=.01,confidence=.95,shots=512,
                                     max_shots=20000,max_queries=10000000,max_power=256)
        # Independent recurrence and analytic identity are strictly posthoc.
        from reference_solver import reference
        ref=reference(cfg,3);p=ref['rows'][-1]['probabilities'][1];theta=math.asin(math.sqrt(p))
        for row in engine.evolutions:
            row['posthoc_amplification_error']=abs(row['actual_gate_probability']-math.sin((2*row['power']+1)*theta)**2)
        interval=result['confidence_interval'];estimate=result['estimate']
        final={**base,'status':result['status'],'estimator':result,'measurement_batches':batches,
            'actual_evolutions':engine.evolutions,'actual_primitive_gate_applications':engine.used,
            'peak_basis':engine.peak,'unfinished_power':engine.active,'A_inverse_state_error_squared':inverse_error,
            'posthoc_reference':ref,'posthoc_reference_p_D':p,
            'posthoc_relative_error':None if estimate is None else abs(estimate-p)/p,
            'posthoc_interval_contains_reference':None if interval is None else interval[0]<=p<=interval[1],
            'seconds':time.perf_counter()-begin,'process_cpu_seconds':time.process_time()-cpu,
            'numerical_scope':'Unpruned complex128 actual gates; empirical inverse/norm/work/formula checks, no rigorous accumulated roundoff enclosure.',
            'reused_prefix_cost_not_discounted_from_estimator_queries':True}
        write_new(out/'result.json',final);progress(result['status'])
        print(json.dumps({'status':result['status'],'estimate':estimate,'CI':interval,'costs':result['costs']},ensure_ascii=True),flush=True)
        return 0 if result['status']=='completed' else 2
    except BaseException as exc:
        progress('failed',error=repr(exc));write_new(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'actual_primitive_gate_applications':engine.used,'peak_basis':engine.peak,'batches':batches})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','worker'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);args=parser.parse_args();out=args.out.resolve()
    if args.action=='prepare':prepared(out);return 0
    if args.action=='worker':
        if os.environ.get('VRQO_SUPERVISED')!='1':raise ValueError('Use run to retain memory/process guard')
        return worker(out)
    from supervised import run_limited
    result=run_limited([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out)],cwd=ROOT,
        output_dir=out/'supervision',memory_bytes=PROTOCOL['memory_mib']*1024**2,
        threads=1,blas_threads=1,heartbeat_sec=0,resource_sample_sec=1,resource_publish_sec=60,echo_output=True)
    print(json.dumps({'supervision_status':result['status'],'reaped':result['all_processes_reaped']},ensure_ascii=True),flush=True)
    return 0 if result['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
