"""Pure request resolution. No simulation, environment mutation or file writes."""
from __future__ import annotations
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import uuid

TASKS = {'reference', 'screen', 'circuit', 'validate', 'mps_probe', 'mps_batch', 'mc', 'test', 'iqae', 'mlae'}
COMMON = {'task', 'config', 'suite', 'case_id', 'steps', 'profile', 'bond', 'mps_trunc',
          'cpu_budget', 'aer_threads', 'mps_threads', 'memory_mib', 'counts_check',
          'shots', 'histories', 'seed', 'outdir', 'dry_run', 'mps_lapack', 'blas_threads',
          'perf_prefix_gates', 'perf_reference', 'batch_workers', 'batch_memory_mib',
          'relative_error', 'confidence', 'iqae_max_shots', 'iqae_max_queries', 'iqae_max_grover_power'}
DIAGNOSTIC_FIELDS = {'iqae_diagnostic', 'iqae_powers', 'iqae_max_sparse_states',
                     'iqae_max_compiled_gates', 'iqae_compare_to'}
COMMON.update(DIAGNOSTIC_FIELDS)
APPLICABLE = {
    'suite': {'screen', 'validate', 'mps_probe'},
    'case_id': {'screen', 'validate', 'mps_probe'},
    'profile': {'circuit', 'validate', 'mps_probe', 'test', 'iqae', 'mlae'},
    'bond': {'mps_probe', 'test', 'iqae', 'mlae'},
    'mps_trunc': {'mps_probe', 'test', 'iqae', 'mlae'},
    'mps_lapack': {'mps_probe'}, 'counts_check': {'mps_probe'},
    'shots': {'mps_probe', 'test', 'iqae', 'mlae'},
    'histories': {'mc'},
    'seed': {'mc', 'mps_probe', 'test', 'iqae', 'mlae'},
    'aer_threads': {'mps_probe', 'test', 'iqae', 'mlae'},
    'mps_threads': {'mps_probe', 'test', 'iqae', 'mlae'},
    'blas_threads': {'mps_probe', 'iqae', 'test'},
    'perf_prefix_gates': {'mps_probe'}, 'perf_reference': {'mps_probe'},
}
APPLICABLE['batch_workers'] = {'mps_batch'}
APPLICABLE['batch_memory_mib'] = {'mps_batch'}
for _key in ('relative_error','confidence','iqae_max_shots','iqae_max_queries','iqae_max_grover_power'):
    APPLICABLE[_key] = {'iqae'}
APPLICABLE['iqae_max_grover_power'].add('mlae')
for _key in DIAGNOSTIC_FIELDS:
    APPLICABLE[_key] = {'iqae'}
for _key in ('suite','profile','bond','mps_trunc','mps_lapack','counts_check','shots','seed',
             'aer_threads','mps_threads','blas_threads'):
    APPLICABLE[_key].add('mps_batch')


def _integer(value, field, minimum=1):
    if isinstance(value, str):
        try: value = int(value)
        except ValueError: raise ValueError(f'{field} must be an integer') from None
    if type(value) is not int or value < minimum:
        raise ValueError(f'{field} must be an integer >= {minimum}')
    return value


def _boolean(value, field):
    if type(value) is bool: return value
    if type(value) is int and value in (0, 1): return bool(value)
    if isinstance(value, str) and value.lower() in ('true', 'false', '0', '1'):
        return value.lower() in ('true', '1')
    raise ValueError(f'{field} must be true/false or 0/1')


def resolve_run_request(request: dict, *, project_root: Path) -> dict:
    """Resolve CLI > CMD > selected suite/model > defaults without side effects.

    Flat keys are explicit CLI/API values. Nested cmd/cli allow caller provenance.
    None/empty strings mean unspecified; zero and False remain explicit.
    """
    from models import model_from_dict, document_info
    from config import default_config
    root = Path(project_root).resolve()
    if not isinstance(request, dict): raise ValueError('request must be a dict')
    if set(request) - COMMON - {'cli', 'cmd'}:
        raise ValueError(f'Unknown request fields: {sorted(set(request)-COMMON-{"cli", "cmd"})}')
    values, origins, overrides = {}, {}, []
    for origin, layer in (('cmd', request.get('cmd', {})),
                          ('cli', {k:v for k,v in request.items() if k in COMMON}),
                          ('cli', request.get('cli', {}))):
        if not isinstance(layer, dict) or set(layer)-COMMON:
            raise ValueError('Unknown fields or invalid request layer')
        for key, value in layer.items():
            if value is None or value == '': continue
            if key in values and values[key] != value:
                overrides.append(dict(field=key, previous=values[key], value=value, source=origin))
            values[key], origins[key] = value, origin
    task = values.get('task', 'mc')
    if task not in TASKS: raise ValueError(f'Unknown task: {task}')
    if task == 'mps_batch':
        from batch_runtime import resolve_batch_request
        result = resolve_batch_request(values, project_root=root)
        result.update(provenance=origins, overrides=overrides,
            precedence='explicit CLI > explicit CMD > explicit batch task > default')
        return result
    if values.get('config') and values.get('suite'): raise ValueError('CONFIG and SUITE are mutually exclusive')
    if task=='reference' and not values.get('config'):
        raise ValueError('reference requires explicit CONFIG; legacy-reference retains the historical default model')
    for key in values:
        if key in APPLICABLE and task not in APPLICABLE[key]:
            raise ValueError(f'{key} is inapplicable to {task}')
    if task in ('screen', 'validate', 'mps_probe') and not (values.get('suite') or values.get('config')):
        raise ValueError(f'{task} requires CONFIG or SUITE')
    def path(value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (root/p).resolve()
    def document(value):
        info=document_info(value,project_root=root)
        return info['locator'],info['document'],info['sha256']
    spec = None
    cases = []
    if values.get('suite'):
        sp, spec, _ = document(values['suite'])
        values['suite'] = str(sp)
        list_key = 'candidates' if task == 'screen' else 'cases'
        source_cases = spec.get(list_key, [])
        ids = [c['id'] for c in source_cases]
        if len(ids) != len(set(ids)): raise ValueError('Duplicate case IDs')
        if any(not isinstance(i,str) or not i or i in ('.','..') or '/' in i or '\\' in i or ':' in i for i in ids):
            raise ValueError('Case IDs must be safe single filenames')
        if not source_cases and spec.get('kind') not in ('precision_audit', 'svd_calibration'):
            raise ValueError('suite must contain cases')
        selected = values.get('case_id')
        if selected and selected not in ids: raise ValueError('Unknown CASE_ID / case_id')
        scientific_overrides = {'steps', 'profile', 'bond', 'mps_trunc', 'mps_lapack', 'seed', 'shots', 'counts_check'} & values.keys()
        if len(source_cases) > 1 and scientific_overrides and not selected:
            raise ValueError('Multi-case overrides require CASE_ID / case_id')
        if not source_cases and scientific_overrides:
            raise ValueError('Case overrides are inapplicable to this diagnostic suite')
        chosen = [copy.deepcopy(c) for c in source_cases if not selected or c['id'] == selected]
        for c in chosen:
            cp, doc, config_hash = document(c['config'])
            model_from_dict(doc)  # Validate exact document before dispatch.
            c['config'] = str(cp)
            steps_key = 'max_steps' if task == 'screen' else 'steps'
            steps = c.get(steps_key, spec.get(steps_key, doc.get('steps', 15)))
            if 'steps' in values:
                overrides.append(dict(field=f"{c['id']}.{steps_key}", previous=steps, value=values['steps'], source=origins['steps']))
                steps = values['steps']
            c[steps_key] = _integer(steps, 'steps')
            if 'profile' in values:
                if task == 'validate': c['profiles'] = [values['profile']]
                else: c['profile'] = values['profile']
            cases.append(dict(c, steps=c[steps_key], model_document=doc,
                              config_sha256=config_hash))
        if source_cases: spec[list_key] = chosen
    else:
        if values.get('case_id'): raise ValueError('case_id requires a suite')
        if values.get('config'):
            cp, doc, config_hash = document(values['config'])
            model_from_dict(doc)
            values['config'] = str(cp)
        else:
            doc, config_hash = asdict(default_config()), None
        cases = [dict(id='model', config=values.get('config'),
                      steps=_integer(values.get('steps', doc.get('steps', 15)), 'steps'),
                      profile=values.get('profile', 'baseline'), model_document=doc, config_sha256=config_hash)]
        if task in ('screen','validate','mps_probe'):
            c={k:v for k,v in cases[0].items() if k not in ('model_document','config_sha256')}
            if task=='screen':
                c['max_steps']=c.pop('steps')
                spec={'candidates':[c]}
            else:
                c['profiles']=[c['profile']]
                spec={'cases':[c], 'bond_dimensions':[32], 'truncation_thresholds':[1e-16]}
    defaults = dict(task=task, config=None, suite=None, case_id=None, steps=15, profile='baseline',
                    bond=None, mps_trunc=None, mps_lapack=None, cpu_budget=4, aer_threads=1,
                    mps_threads=1, blas_threads=1, perf_prefix_gates=None, perf_reference=None,
                    memory_mib=None, counts_check=True, shots=20000,
                    seed=1234, outdir=str(root/'outputs'/f'run_{task}_{uuid.uuid4().hex[:12]}'), dry_run=False)
    if task=='iqae':
        defaults.update(relative_error=.01,confidence=.95,iqae_max_shots=200000,
            iqae_max_queries=100000000000,iqae_max_grover_power=10000000,shots=512,
            iqae_diagnostic=None,iqae_powers=None,iqae_max_sparse_states=60000,
            iqae_max_compiled_gates=4000000,iqae_compare_to=None)
    if task=='mc':
        defaults['histories']=200000
    if spec is not None:
        defaults.update(seed=spec.get('seed', 20261030), shots=spec.get('shots',256),
                        counts_check=spec.get('counts_check',True))
    if len(cases)==1:
        defaults['steps']=cases[0]['steps']
        defaults['profile']=cases[0].get('profile','baseline')
    resolved = {**defaults, **values}
    for key in defaults:
        if key not in origins:
            origins[key] = 'suite' if spec is not None and key in ('steps','profile','bond','mps_trunc','shots','seed','counts_check') else 'model' if key == 'steps' and values.get('config') else 'default'
    for key in ('cpu_budget','aer_threads','mps_threads','blas_threads','steps','shots'):
        resolved[key] = _integer(resolved[key],key)
    resolved['seed'] = _integer(resolved['seed'],'seed',0)
    if task=='mc':resolved['histories']=_integer(resolved['histories'],'histories')
    if task=='iqae':
        for key in ('relative_error','confidence'):
            value=resolved[key]
            if isinstance(value,bool):raise ValueError(f'{key} must be finite and strictly between 0 and 1')
            value=float(value)
            if not math.isfinite(value) or not 0<value<1:
                raise ValueError(f'{key} must be finite and strictly between 0 and 1')
            resolved[key]=value
        for key in ('iqae_max_shots','iqae_max_queries','iqae_max_grover_power'):
            resolved[key]=_integer(resolved[key],key,0 if key=='iqae_max_grover_power' else 1)
        mode=resolved['iqae_diagnostic']
        if mode is not None:
            if mode not in ('compile','probability','sparse'):
                raise ValueError('iqae_diagnostic must be compile, probability or sparse')
            raw=resolved['iqae_powers']
            if isinstance(raw,str): raw=raw.split(',')
            if not isinstance(raw,(list,tuple)) or not raw:
                raise ValueError('Fixed-power diagnostics require explicit iqae_powers')
            powers=[_integer(k,'iqae_powers',0) for k in raw]
            if len(powers)!=len(set(powers)):
                raise ValueError('iqae_powers must not contain duplicates')
            if max(powers)>resolved['iqae_max_grover_power']:
                raise ValueError('iqae_powers exceeds iqae_max_grover_power')
            resolved['iqae_powers']=powers
            for key in ('iqae_max_sparse_states','iqae_max_compiled_gates'):
                resolved[key]=_integer(resolved[key],key)
            if resolved['iqae_compare_to']:
                resolved['iqae_compare_to']=str(path(resolved['iqae_compare_to']))
                if not Path(resolved['iqae_compare_to']).is_dir():
                    raise ValueError('iqae_compare_to must be an existing export directory')
        elif any(key in values for key in DIAGNOSTIC_FIELDS-{'iqae_diagnostic'}):
            raise ValueError('Fixed-power options require iqae_diagnostic')
    for key in ('bond', 'memory_mib', 'perf_prefix_gates'):
        if resolved[key] is not None: resolved[key] = _integer(resolved[key],key)
    if resolved['mps_trunc'] is not None:
        v=resolved['mps_trunc']
        if isinstance(v,bool): raise ValueError('mps_trunc must be finite nonnegative')
        v=float(v)
        if not math.isfinite(v) or v<0: raise ValueError('mps_trunc must be finite nonnegative')
        resolved['mps_trunc']=v
    for key in ('dry_run','counts_check'):
        resolved[key]=_boolean(resolved[key],key)
    if resolved['mps_lapack'] is not None: resolved['mps_lapack']=_boolean(resolved['mps_lapack'],'mps_lapack')
    if resolved['profile'] not in ('compressed','baseline','full_history','manual','default_light','light_baseline'):
        raise ValueError('Unknown kernel profile')
    if resolved['aer_threads']*resolved['mps_threads']*resolved['blas_threads']>resolved['cpu_budget']:
        raise ValueError('Aer * MPS * BLAS threads exceeds requested CPU budget')
    if resolved['perf_reference'] and not resolved['perf_prefix_gates']:
        raise ValueError('perf_reference requires perf_prefix_gates')
    if resolved['perf_prefix_gates']:
        if len(cases)!=1 or resolved['counts_check']:
            raise ValueError('Performance prefix requires one case and counts_check=false')
        if resolved['perf_reference']:resolved['perf_reference']=str(path(resolved['perf_reference']))
    resolved['outdir']=str(path(resolved['outdir']))
    backend = {}
    if spec is not None and task == 'mps_probe':
        for key, dest in (('bond','bond_dimensions'),('mps_trunc','truncation_thresholds')):
            if key in values:
                overrides.append(dict(field=dest,previous=spec.get(dest),value=[resolved[key]],source=origins[key]))
                spec[dest]=[resolved[key]]
        spec['seed'],spec['shots'],spec['counts_check']=resolved['seed'],resolved['shots'],resolved['counts_check']
        if 'mps_lapack' in values:
            spec.setdefault('backend_options',{})['mps_lapack']=resolved['mps_lapack']
            for c in spec.get('cases',[]): c.get('backend_options',{}).pop('mps_lapack',None)
        backend={k:copy.deepcopy(spec.get(k)) for k in ('bond_dimensions','truncation_thresholds','backend_options','seed','shots','counts_check')}
        # Reuse validation without importing Qiskit: invalid suite settings must fail in dry-run.
        from validation import _probe_settings, _probe_backend_options
        bonds, thresholds, _ = _probe_settings(spec)
        if resolved['perf_prefix_gates'] and (len(bonds)!=1 or len(thresholds)!=1):
            raise ValueError('Performance prefix requires a single bond and truncation setting')
        backend['per_case_options']=[]
        for c in spec['cases']:
            extras={**spec.get('backend_options',{}),**c.get('backend_options',{})}
            options=_probe_backend_options(resolved['cpu_budget'],resolved['aer_threads'],resolved['mps_threads'],extras,bonds[0],thresholds[0],seed=resolved['seed'])
            backend['per_case_options'].append(dict(case_id=c['id'],first_setting=options))
    resources={k:resolved[k] for k in ('cpu_budget','aer_threads','mps_threads','blas_threads','memory_mib')}
    precision_contract = (dict(scope='single_case_complete_estimation',
        relative_error=resolved['relative_error'],confidence=resolved['confidence'],
        failure_probability=1-resolved['confidence'],
        stop_rule='valid confidence interval [L,U], L>0 and (U-L)/(2*L)<=relative_error; estimate=(L+U)/2',
        budget_exhaustion='budget_within_target_not_achieved',
        numerical_parameters='bond and mps_trunc control simulator error separately; statistical precision does not certify total error',
        reference_used_for_stopping=False) if task=='iqae' else
        dict(scope='direct_MPS_numerical_response',relative_error=.01,
        reference='independent same-model same-T',statistical_estimation=False) if task=='mps_probe' else None)
    return dict(schema_version=1, task=task, counts_check=resolved['counts_check'],
                dry_run=resolved['dry_run'],outdir=resolved['outdir'],backend_options=backend,
                arguments=resolved, cases=cases, effective_suite=spec,
                backend=backend, resources=resources, provenance=origins, overrides=overrides,
                precision_contract=precision_contract,
                diagnostic_contract=(dict(mode=resolved['iqae_diagnostic'],
                    powers=resolved['iqae_powers'],estimator_executed=False,
                    sampling_executed=False,
                    objective_probability_only=resolved['iqae_diagnostic']=='probability',
                    compiler='historical measured-circuit transpile; no component substitution',
                    scope='fixed-power circuit identity or numerical diagnostic; not a complete IQAE estimate')
                    if task=='iqae' and resolved['iqae_diagnostic'] else None),
                precedence='explicit CLI > explicit CMD > selected suite/model > default',
                simulation_executed=False)
