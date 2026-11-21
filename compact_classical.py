"""Compact-slab declarations and an independent Decimal verification recurrence.

This is a diagnostic library, not a second production CLI. Ordinary screening
and float64 reference runs continue through main.py. Decimal transport does not
call production geometry, CSR reference helpers, or any quantum code.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict,is_dataclass
from decimal import Decimal,localcontext
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
DIRECTIONS=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))


def build_compact(base_cells,tau,omega,factor=1):
    if type(base_cells) is not int or base_cells<=0 or type(factor) is not int or factor<=0:
        raise ValueError('cell count and refinement factor must be positive integers')
    if not math.isfinite(float(tau)) or float(tau)<=0 or not 0<=float(omega)<=1:
        raise ValueError('positive finite tau and omega in [0,1] required')
    from presets import get_preset
    c=get_preset('uniform_tau2_w5')
    c.pop('metadata',None);g=c['geom'];g['slab_x1']=g['slab_x0']+base_cells-1
    g['detector_x']=g['slab_x1']+3;g['nx']=g['detector_x']+3;c['layers'][0]['x1']=g['slab_x1']
    sigma=float(tau)/(base_cells*c['h'])
    if factor!=1:
        for k in ('nx','ny','nz'):g[k]=factor*(g[k]-1)+1
        for k in ('slab_x0','detector_x','duct_y0','duct_y1','duct_z0','duct_z1','roi_y0','roi_y1','roi_z0','roi_z1'):g[k]*=factor
        g['slab_x1']=factor*(g['slab_x1']+1)-1
        for layer in c['layers']:layer['x0']*=factor;layer['x1']=factor*(layer['x1']+1)-1
        for k in ('x0','y0','z0'):c['source'][k]*=factor
        c['h']/=factor
    c['mats']['tungsten']['p_int']=[-math.expm1(-sigma*c['h'])]
    c['mats']['tungsten']['p_abs']=[1-float(omega)]
    c['metadata']=dict(base_cells=base_cells,refinement_factor=factor,tau=str(tau),omega=str(omega),
        Sigma_t=sigma,eta=float(tau)/(base_cells*factor),physical_length=base_cells*.5,
        parameter_interpretation='dimensionless absorption-dominated mechanism; uncalibrated material slots',
        physical_window='independent T chosen at each h until p_S/p_D <=1e-4; near-final response comparison',
        mapping='interfaces/source/detector support exactly aligned; no rounding; fixed 1 u^2 closed acceptance square',
        full_geometric_detector_area=1,source_normalization=1,source_front_gap=1,rear_detector_gap=1)
    return c


def decimal_reference(cfg,steps,digits=50,coefficient_mode='physical'):
    """Independently recurse with Decimal coefficients and positive D inflows.

    physical: generate exp(-tau/N) and omega from original decimal metadata;
    fixed_float: lift stored float64 p_int/p_abs exactly using from_float.
    Neither mode imports or calls the float64 reference or production geometry.
    """
    if isinstance(steps,bool) or not isinstance(steps,Integral) or steps<0:raise ValueError('nonnegative integer steps required')
    if type(digits) is not int or digits<30:raise ValueError('digits must be integer >=30')
    if coefficient_mode not in ('physical','fixed_float'):raise ValueError('unknown coefficient mode')
    c=asdict(cfg) if is_dataclass(cfg) else deepcopy(cfg)
    if c.get('model')!='uniform_slab' or c['groups']['n_groups']!=1 or len(c['layers'])!=1:
        raise ValueError('Decimal diagnostic supports declared one-group uniform slabs only')
    if any(v for m in c['mats'].values() for v in m['q_down']):raise ValueError('downscatter not supported in compact diagnostic')
    g=c['geom'];s=c['source'];layer=c['layers'][0];start=time.perf_counter()
    with localcontext() as ctx:
        ctx.prec=digits;one=Decimal(1);zero=Decimal(0)
        if coefficient_mode=='physical':
            meta=c['metadata'];tau=Decimal(str(meta['tau']));omega=Decimal(str(meta['omega']))
            cells=Decimal(layer['x1']-layer['x0']+1)
            nc=(-tau/cells).exp();pi=one-nc;absorb=pi*(one-omega);scatter=pi*omega/6
        else:
            m=c['mats'][layer['material']]
            pi=Decimal.from_float(float(m['p_int'][0]));pa=Decimal.from_float(float(m['p_abs'][0]))
            nc=one-pi;absorb=pi*pa;scatter=pi*(one-pa)/6
        coefficients=(nc,absorb,scatter)
        # Outside the slab the declaration must be vacuum in both modes.
        if c['mats']['concrete']['p_int'][0]!=0:raise ValueError('compact exterior must be vacuum')
        live={(s['x0'],s['y0'],s['z0'],s.get('dir0',0)):one}
        terminal=[zero,zero,zero];rows=[];peak=1;minimum=one
        for t in range(1,steps+1):
            nxt=defaultdict(Decimal);first_D=zero
            for (x,y,z,d),mass in live.items():
                if x==0 or x==g['nx']-1 or y==0 or y==g['ny']-1 or z==0 or z==g['nz']-1:
                    terminal[2]+=mass;continue
                if x==g['detector_x'] and g['roi_y0']<=y<=g['roi_y1'] and g['roi_z0']<=z<=g['roi_z1']:
                    terminal[0]+=mass;first_D+=mass;continue
                trans,ab,sc=coefficients if layer['x0']<=x<=layer['x1'] else (one,zero,zero)
                terminal[1]+=mass*ab
                if trans:
                    dx,dy,dz=DIRECTIONS[d];nxt[(x+dx,y+dy,z+dz,d)]+=mass*trans
                if sc:
                    part=mass*sc
                    for dn,(dx,dy,dz) in enumerate(DIRECTIONS):nxt[(x+dx,y+dy,z+dz,dn)]+=part
            live=dict(nxt);survival=sum(live.values(),zero);peak=max(peak,len(live))
            if live:minimum=min(minimum,min(v for v in live.values() if v))
            rows.append(dict(steps=t,p_D=str(terminal[0]),p_A=str(terminal[1]),p_B=str(terminal[2]),p_S=str(survival),first_D=str(first_D)))
        survival=sum(live.values(),zero)
        stringify=lambda x:'0' if x==0 else str(x)
        return dict(p_D=stringify(terminal[0]),p_A=stringify(terminal[1]),p_B=stringify(terminal[2]),p_S=stringify(survival),
            rows=rows,steps=steps,digits=digits,coefficient_mode=coefficient_mode,
            coefficient_meaning='physical tau/omega decimal regenerated' if coefficient_mode=='physical' else 'exact binary64 p_int/p_abs lifted by Decimal.from_float; recurrence then evaluated at requested precision',
            coefficients=dict(no_collision=str(nc),absorb=str(absorb),scatter_per_direction=str(scatter)),
            mass_residual=str(abs(sum(terminal,zero)+survival-one)),peak_live_states=peak,
            minimum_positive_live_probability=str(minimum),seconds=time.perf_counter()-start,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def audit_precision(config_paths,steps_by_name):
    """Return the 50/80-digit physical vs fixed-binary64 comparison table."""
    from models import load_document,load_model
    from reference_solver import reference
    import numpy as np
    rows=[]
    for path in config_paths:
        path=Path(path);c=load_document(path);steps=steps_by_name[path.stem]
        fp=reference(load_model(path),steps)['rows'][-1]['p_D']
        runs={f'{mode}_{digits}':decimal_reference(c,steps,digits,mode)
              for mode in ('physical','fixed_float') for digits in (50,80)}
        with localcontext() as ctx:
            ctx.prec=80
            phys=Decimal(runs['physical_80']['p_D']);fixed=Decimal(runs['fixed_float_80']['p_D']);v=Decimal.from_float(fp)
            eta=Decimal(c['metadata']['tau'])/Decimal(c['metadata']['base_cells']*c['metadata']['refinement_factor'])
            ideal_nc=(-eta).exp();stored_pi=float(c['mats']['tungsten']['p_int'][0]);actual_nc=Decimal.from_float(1-stored_pi)
            row=dict(id=path.stem,steps=steps,p_D_float64=fp,runs=runs,
                relative_float64_vs_physical80=str(abs(v/phys-1)),relative_float64_vs_fixed80=str(abs(v/fixed-1)) if fixed else None,
                relative_fixed80_vs_physical80=str(abs(fixed/phys-1)),
                relative_physical50_vs80=str(abs(Decimal(runs['physical_50']['p_D'])/phys-1)),
                relative_fixed50_vs80=str(abs(Decimal(runs['fixed_float_50']['p_D'])/fixed-1)) if fixed else None,
                no_collision_subtraction_relative_error=str(abs(actual_nc/ideal_nc-1)),
                p_int_float=stored_pi,one_minus_p_int_float=1-stored_pi,
                target_relative=1e-8,passes_numeric_target=abs(v/phys-1)<=Decimal('1e-8'))
        rows.append(row)
    return dict(rows=rows,finfo={name:{'bits':np.finfo(dt).bits,'nmant':np.finfo(dt).nmant,'eps':str(np.finfo(dt).eps),'tiny':str(np.finfo(dt).tiny)}
                    for name,dt in [('float64',np.float64),('longdouble',np.longdouble)]},
                interpretation='positive detector inflow summation; numerical comparison, not a rigorous interval error bound')
