"""Finite, predeclared classical screening with a single-thread CPU allocation."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import time
from contextlib import nullcontext

from reference_solver import reference

ROOT=Path(__file__).resolve().parent


def _write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)


def _positive_integer(value,name):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<=0:
        raise ValueError(f'{name} must be a positive integer (not bool or float)')
    return int(value)


def run_screen(suite_path, outdir, cpu_budget=4):
    from models import load_model, document_info
    from runtime_identity import validate_output_path
    cpu_budget=_positive_integer(cpu_budget,'cpu_budget')
    suite_info=document_info(suite_path)
    suite=suite_info['document']
    _positive_integer(suite.get('max_steps',512),'max_steps')
    for entry in suite['candidates']:
        _positive_integer(entry.get('max_steps',suite.get('max_steps',512)),'candidate max_steps')
    out=validate_output_path(outdir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'screening_manifest.json').exists():raise FileExistsError(out/'screening_manifest.json')
    candidates=suite['candidates'];ids=[c['id'] for c in candidates]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate candidate id')
    if any(Path(i).name!=i or i in ('.','..') for i in ids):raise ValueError('invalid candidate id')
    manifest=dict(suite=suite,suite_sha256=suite_info['sha256'],
        cpu_budget=int(cpu_budget),workers=1,threads_per_worker=1,
        policy='finite physical step budget; no wall-clock cutoff; results never overwritten')
    _write(out/'screening_manifest.json',manifest)
    classical=out/'classical';classical.mkdir(exist_ok=True)
    summaries=[];start=time.perf_counter()
    with nullcontext():
        manifest['effective_threadpools']='single-thread scipy CSR and NumPy elementwise reductions; no dense BLAS operations'
        _write(out/'screening_resources.json',manifest)
        for entry in candidates:
            info=document_info(entry['config']);cfg=info['document']
            validated=load_model(entry['config'])
            limit=_positive_integer(entry.get('max_steps',suite.get('max_steps',512)),'max_steps')
            # Refined cases use an exactly matched coarse physical observation T*h.
            baseline=entry.get('match_observation_to')
            if baseline:
                previous=next(s for s in summaries if s['id']==baseline)
                mapped_steps=previous['T']*previous['h']/validated.h
                if not math.isfinite(mapped_steps) or not math.isclose(mapped_steps,round(mapped_steps),rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('matched physical observation window is not an integer number of target-grid steps')
                limit=_positive_integer(round(mapped_steps),'matched observation steps')
            try:
                r=reference(validated,limit,tail_relative_tolerance=None if baseline else suite.get('tail_relative_tolerance',.01))
                r['candidate']=entry;r['config_file_sha256']=info['file_sha256'];r['config_document_sha256']=info['sha256']
                _write(classical/(entry['id']+'.json'),r)
                last=r['rows'][-1];d=r['diagnostics'];pd=last['p_D'];meta=cfg.get('metadata',{})
                row=dict(id=entry['id'],model=cfg.get('model','legacy_duct'),role=entry.get('role','candidate'),
                    config=entry['config'],T=last['steps'],h=cfg.get('h',1),p_D=pd,p_A=last['p_A'],p_B=last['p_B'],p_S=last['p_S'],
                    probability_sum=last['probability_sum'],mass_balance_error=max(v['norm_error'] for v in r['rows']),
                    p_D_uncollided=d['p_D_uncollided'],p_D_scattered=d['p_D_scattered'],
                    scattered_fraction=d['p_D_scattered']/pd if pd else None,
                    earliest_detection_step=d['earliest_detection_step'],
                    detected_mean_scatter_count=d['detected_mean_scatter_count'],
                    leakage_front=d['leakage']['front'],leakage_side=d['leakage']['side'],leakage_rear=d['leakage']['rear'],
                    straight_tau=d['straight_effective_optical_thickness'],reachable_states=d['reachable_states'],
                    transition_nnz=d['transition_nnz'],max_live_states=d['max_live_states'],
                    elapsed_seconds=d['total_seconds'],process_peak_rss_bytes=d['process_peak_rss_bytes'],
                    tail_relative_bound=last['tail_bound_relative_to_current_D'],
                    config_sha256=r['identity']['config_sha256'],reference_source_sha256=r['identity']['reference_source_sha256'],
                    status='complete',physics=meta)
                for eps,label in ((.01,'1pct'),(.05,'5pct')):
                    qualifying=[v['steps'] for v in r['rows'] if v['p_D']>0 and v['p_S']<=eps*v['p_D']]
                    row['T_tail_'+label]=qualifying[0] if qualifying else None
                    row['tail_'+label+'_met']=bool(qualifying)
                    row['approx_95pct_sampling_histories_'+label]=int(__import__('math').ceil(1.95996398454**2*(1-pd)/(eps**2*pd))) if pd else None
                summaries.append(row)
                print(f"screen {entry['id']}: T={last['steps']}, D={pd:.10g}, S={last['p_S']:.3g}, {d['total_seconds']:.3f}s",flush=True)
            except Exception as exc:
                failure=dict(candidate=entry,status='failed',exception=type(exc).__name__,message=str(exc))
                _write(classical/(entry['id']+'.failed.json'),failure);summaries.append(dict(id=entry['id'],status='failed'))
                raise
    result=dict(candidates=summaries,total_seconds=time.perf_counter()-start,
        notes=['p_D is finite-T first ideal-plane hit probability.',
            'Sampling estimates are single Bernoulli-observable normal approximations at 95%; no simultaneous-coverage correction or AE claim.',
            'No MPS feasibility inference follows from classical runtime or state count.'])
    _write(out/'screening_summary.json',result)
    fields=[k for k in summaries[0] if k!='physics']
    with (out/'screening_summary.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    return result
