"""Independent float64 discrete-transport reference; no production physics imports.

Only the declarative dataclass/dictionary is shared. Coordinates, material
selection, event order and transition construction are implemented here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix

DIRS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
ROOT = Path(__file__).resolve().parent


def _plain(cfg):
    c=asdict(cfg) if is_dataclass(cfg) else json.loads(json.dumps(cfg))
    c.pop('metadata',None)
    return c


def _material(c, x, y, z):
    g=c['geom']
    if c.get('model','legacy_duct') == 'legacy_duct':
        name='concrete'
        if g['slab_x0']<=x<=g['slab_x1']:
            name='air' if (g['duct_y0']<=y<=g['duct_y1'] and
                           g['duct_z0']<=z<=g['duct_z1']) else 'tungsten'
    else:
        name='concrete'
        for layer in c.get('layers',[]):
            if layer['x0']<=x<=layer['x1']:
                name=layer['material'];break
    return name,c['mats'][name]


def _peak_rss():
    try:
        import psutil
        mem=psutil.Process().memory_info()
        return int(getattr(mem,'peak_wset',mem.rss))
    except (ImportError,OSError):
        return None


def _identity(c, steps):
    from runtime_identity import make_identity,digest
    canonical=json.dumps(c,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result=make_identity(config=c,profile='independent_reference',compile_options=None,
        measurement='deterministic complete classical distribution',backend={'algorithm':'CSR','dtype':'float64'},
        phase='reference',extra={'steps':steps})
    result.update(config_sha256=hashlib.sha256(canonical).hexdigest(),
                reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                steps=steps,algorithm='reachable CSR deterministic recursion',dtype='float64')
    result['identity_hash']=digest({k:v for k,v in result.items() if k!='identity_hash'})
    return result


def _build(c):
    g=c['geom'];s=c['source'];ng=c['groups']['n_groups']
    first=(s['x0'],s['y0'],s['z0'],s.get('dir0',0),s['g0'])
    states=[first];index={first:0};row=[];col=[];sc=[];nc=[]
    terminal=[];leaks=[];abs_material=[]
    for i,(x,y,z,d,e) in enumerate(states):
        if x in (0,g['nx']-1) or y in (0,g['ny']-1) or z in (0,g['nz']-1):
            terminal.append((0.,0.,1.));abs_material.append(None)
            # Exclusive classification: x faces have priority at edges/corners.
            leaks.append('front' if x==0 else 'rear' if x==g['nx']-1 else 'side')
            continue
        if x==g['detector_x'] and g['roi_y0']<=y<=g['roi_y1'] and g['roi_z0']<=z<=g['roi_z1']:
            terminal.append((1.,0.,0.));leaks.append(None);abs_material.append(None);continue
        name,m=_material(c,x,y,z)
        pi=float(m['p_int'][e]);pa=float(m['p_abs'][e]);q=float(m['q_down'][e]) if e+1<ng else 0.
        if not all(math.isfinite(p) and 0<=p<=1 for p in (pi,pa,q)):
            raise ValueError('reference material probabilities must lie in [0,1]')
        terminal.append((0.,pi*pa,0.));leaks.append(None);abs_material.append(name)
        local=defaultdict(lambda:[0.,0.])
        dx,dy,dz=DIRS[d]
        local[(x+dx,y+dy,z+dz,d,e)][0]+=1-pi
        for dn,(dx,dy,dz) in enumerate(DIRS):
            mass=pi*(1-pa)/6
            local[(x+dx,y+dy,z+dz,dn,e)][1]+=mass*(1-q)
            if q:local[(x+dx,y+dy,z+dz,dn,e+1)][1]+=mass*q
        for state,(a,b) in local.items():
            if a+b==0:continue
            if state not in index:index[state]=len(states);states.append(state)
            row.append(index[state]);col.append(i);nc.append(a);sc.append(b)
    shape=(len(states),len(states))
    zero=coo_matrix((nc,(row,col)),shape=shape).tocsr();zero.eliminate_zeros()
    scatter=coo_matrix((sc,(row,col)),shape=shape).tocsr();scatter.eliminate_zeros()
    kernel=zero+scatter
    rates=np.asarray(terminal,dtype=float).T
    mass_error=float(np.max(np.abs(np.asarray(kernel.sum(axis=0)).ravel()+rates.sum(axis=0)-1)))
    return states,kernel,zero,scatter,rates,leaks,abs_material,mass_error


def _straight_tau(c):
    s=c['source'];g=c['geom']
    if s.get('dir0',0)!=0 or not (g['roi_y0']<=s['y0']<=g['roi_y1'] and g['roi_z0']<=s['z0']<=g['roi_z1']):
        return None
    value=0.
    for x in range(s['x0'],g['detector_x']):
        _,m=_material(c,x,s['y0'],s['z0']);p=m['p_int'][s['g0']]
        if p==1:return 'infinite'
        value-=math.log1p(-p)
    return value


def reference(cfg, steps, retain_joint=False, *, tail_relative_tolerance=None, min_steps=0):
    """Return all T checkpoints and optional joint keys (status,x,y,z,dir,g).

    Default executes exactly ``steps`` kernels. Screening may stop sooner once
    p_S <= tolerance*p_D, after min_steps. This is a physical step budget,
    never a wall-clock deadline. All sums are floating-point numerical results.
    """
    if isinstance(steps,bool) or not isinstance(steps,Integral) or steps<0:
        raise ValueError('steps must be a nonnegative integer')
    if tail_relative_tolerance is not None and not 0<tail_relative_tolerance<1:
        raise ValueError('tail relative tolerance must be in (0,1)')
    start=time.perf_counter();c=_plain(cfg)
    states,kernel,zero,scatter,rates,leaks,materials,mass_error=_build(c)
    build_seconds=time.perf_counter()-start;n=len(states)
    live=np.zeros(n);live[0]=1;uncollided=live.copy();moment=np.zeros(n)
    total=np.zeros(3);un_D=0.;moment_D=0.;rows=[];first_detection=[]
    leakage={name:0. for name in ('front','side','rear')}
    absorption={name:0. for name in c['mats']}
    leak_masks={name:np.asarray([v==name for v in leaks]) for name in leakage}
    abs_rates={name:rates[1]*np.asarray([v==name for v in materials]) for name in absorption}
    frozen=np.zeros((3,n)) if retain_joint else None
    earliest=None;max_live=1
    for t in range(1,steps+1):
        inc=(rates*live).sum(axis=1);total+=inc
        if inc[0]>0 and earliest is None:earliest=t
        first_detection.append(float(inc[0]))
        un_D+=float((rates[0]*uncollided).sum());moment_D+=float((rates[0]*moment).sum())
        for name,mask in leak_masks.items():leakage[name]+=float(live[mask].sum())
        for name,rate in abs_rates.items():absorption[name]+=float((rate*live).sum())
        if frozen is not None:frozen+=rates*live
        moment=kernel@moment+scatter@live
        uncollided=zero@uncollided;live=kernel@live
        survival=float(live.sum());probs=[survival]+[float(v) for v in total]
        active=int(np.count_nonzero(live));max_live=max(max_live,active)
        rows.append(dict(steps=t,probabilities=probs,p_S=probs[0],p_D=probs[1],
            p_A=probs[2],p_B=probs[3],probability_sum=math.fsum(probs),
            norm_error=abs(math.fsum(probs)-1),first_termination=[0.]+[float(v) for v in inc],
            p_D_uncollided=un_D,p_D_scattered=float(total[0])-un_D,
            tail_bound_absolute=survival,tail_bound_relative_to_current_D=survival/total[0] if total[0]>0 else None,
            live_states=active))
        if tail_relative_tolerance is not None and t>=min_steps and total[0]>0 and survival<=tail_relative_tolerance*total[0]:break
    result=dict(order=['survive','detect','absorb','boundary'],config=c,rows=rows,
        identity=_identity(c,len(rows)),
        diagnostics=dict(earliest_detection_step=earliest,first_detection_by_step=first_detection,
            p_D_uncollided=un_D,p_D_scattered=float(total[0])-un_D,
            detected_mean_scatter_count=moment_D/total[0] if total[0]>0 else None,
            leakage=leakage,absorption_by_material=absorption,
            straight_effective_optical_thickness=_straight_tau(c),reachable_states=n,
            transition_nnz=int(kernel.nnz),max_live_states=max_live,
            one_step_mass_error=mass_error,build_seconds=build_seconds,
            total_seconds=time.perf_counter()-start,process_peak_rss_bytes=_peak_rss(),
            memory_scope='process lifetime high-water RSS on Windows; current RSS fallback elsewhere',
            numerical_status='float64 unsampled reference; tail inequality is analytic, evaluated without interval arithmetic',
            termination_argument='bounded rectangular domain and reachable escape paths; no nonterminating closed class in screening cases',
            tail_method='p_S conservative upper bound; no floating-point linear solve claimed as certification'))
    if retain_joint:
        joint={(0,)+state:float(live[i]) for i,state in enumerate(states) if live[i]!=0}
        for k in range(3):
            joint.update({(k+1,)+state:float(frozen[k,i]) for i,state in enumerate(states) if frozen[k,i]!=0})
        result['joint']=joint
    return result


def run_reference(cfg, steps, outdir):
    """Write a standalone reference result; never overwrite an earlier result."""
    from runtime_identity import validate_output_path
    out=validate_output_path(outdir);out.mkdir(parents=True,exist_ok=True);target=out/'reference.json'
    if target.exists():raise FileExistsError(target)
    result=reference(cfg,steps)
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result
