"""Finite independent transport jobs under one aggregate supervisor.

No MPS trajectory is split, no adaptive AE rounds are scheduled, and no resume
is implicit. Each task invokes the same main.py and retains its own evidence.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parent
RESERVE_MIB = 512


def available_cpus():
    try:return len(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.Error):return psutil.cpu_count() or 1


def resolve_batch_request(values, *, project_root):
    """Resolve already merged CLI/CMD values; reject unsupported controls."""
    from presets import get_preset
    from run_request import resolve_run_request, _integer, _boolean
    allowed={'task','suite','steps','profile','bond','mps_trunc','mps_lapack',
        'cpu_budget','aer_threads','mps_threads','blas_threads','memory_mib',
        'batch_memory_mib','batch_workers','counts_check','shots','seed','outdir','dry_run'}
    values={k:v for k,v in values.items() if v is not None and v!=''}
    if set(values)-allowed:raise ValueError('Unsupported batch controls: '+str(sorted(set(values)-allowed)))
    dry_run=_boolean(values.get('dry_run',False),'dry_run')
    spec=get_preset(values.get('suite','mps_batch_small'))
    if spec.get('kind')!='independent_mps_batch':raise ValueError('Select an independent_mps_batch preset')
    tasks=spec.get('tasks',[])
    ids=[t.get('id') for t in tasks]
    if not ids or len(ids)!=len(set(ids)) or any(not isinstance(i,str) or not i or i in ('.','..') or any(s in i for s in '/\\:') for i in ids):
        raise ValueError('Batch task IDs must be unique safe filenames')
    budget=_integer(values.get('cpu_budget',50),'cpu_budget')
    if budget>60:raise ValueError('Batch CPU budget must not exceed 60')
    cpus=available_cpus()
    host_issues=[]
    if budget>cpus:host_issues.append(f'Batch CPU budget {budget} exceeds available CPU affinity {cpus}')
    workers=_integer(values.get('batch_workers',min(2,len(tasks))),'batch_workers')
    if workers>len(tasks):raise ValueError('batch_workers exceeds the explicit task count')
    memory=_integer(values.get('memory_mib',8192),'memory_mib')
    batch_memory=_integer(values.get('batch_memory_mib',memory*workers+RESERVE_MIB),'batch_memory_mib')
    if memory*workers+RESERVE_MIB>batch_memory:
        raise ValueError('Batch memory must cover all concurrent task limits plus 512 MiB coordinator reserve')
    available_mib=psutil.virtual_memory().available//1024**2
    if batch_memory>available_mib:host_issues.append(f'Batch memory {batch_memory} MiB exceeds currently available {available_mib} MiB')
    if host_issues and not dry_run:raise ValueError('; '.join(host_issues))
    from runtime_identity import digest, source_manifest, software_versions, normalize
    out=Path(values.get('outdir',Path(project_root)/'outputs'/('batch_'+__import__('uuid').uuid4().hex[:12])))
    if not out.is_absolute():out=Path(project_root)/out
    out=out.resolve()
    global_controls={k:v for k,v in values.items() if k in allowed-{'task','suite','cpu_budget','batch_workers','batch_memory_mib','outdir','dry_run'}}
    resolved=[]
    for item in tasks:
        if set(item)-{'id','config','suite','case_id','steps','profile','bond','mps_trunc','mps_lapack','seed'}:
            raise ValueError('Batch preset task contains unsupported fields')
        request=dict(task='mps_probe',aer_threads=1,mps_threads=1,blas_threads=1)
        if not item.get('suite'):
            request.update(counts_check=False,shots=1,seed=20261030,bond=64,mps_trunc=1e-16,mps_lapack=False)
        request.update({k:v for k,v in item.items() if k!='id'})
        request.update(global_controls)
        task_cpu=(_integer(request['aer_threads'],'aer_threads')*
            _integer(request['mps_threads'],'mps_threads')*_integer(request['blas_threads'],'blas_threads'))
        request.update(cpu_budget=task_cpu,memory_mib=memory,outdir=str(out/'tasks'/item['id']))
        r=resolve_run_request(request,project_root=project_root)
        if (len(r['cases'])!=1 or len(r['backend']['bond_dimensions'])!=1 or
                len(r['backend']['truncation_thresholds'])!=1):
            raise ValueError('Each independent batch task must resolve exactly one case, bond and truncation setting')
        resolved.append({'id':item['id'],'request':r})
    demand=sorted([r['request']['resources']['aer_threads']*r['request']['resources']['mps_threads']*
        r['request']['resources']['blas_threads'] for r in resolved],reverse=True)
    concurrent=sum(demand[:workers])
    if concurrent>budget:raise ValueError(f'Concurrent task CPU threads {concurrent} exceeds batch CPU budget {budget}')
    resources=dict(cpu_budget=budget,available_cpus=cpus,batch_workers=workers,
        concurrent_cpu_threads=concurrent,task_memory_mib=memory,batch_memory_mib=batch_memory,
        concurrent_task_memory_mib=memory*workers,coordinator_reserve_mib=RESERVE_MIB,
        available_memory_mib=available_mib,host_fits_request=not host_issues,
        host_resource_issues=host_issues)
    identity={'sources':source_manifest(project_root),'versions':software_versions(),
        'tasks':resolved,'resources':resources,'preset':spec,'outdir':str(out)}
    return normalize(dict(schema_version=1,task='mps_batch',outdir=str(out),tasks=resolved,
        resources=resources,identity=identity,identity_hash=digest(identity),
        dry_run=dry_run,host_fits_request=not host_issues,execution_ready=not host_issues,
        dry_run_policy='Cloud request validation does not reserve resources; execution rechecks host CPU and available memory.',
        simulation_executed=False,
        resume_supported=False,statistical_estimation_status='not_performed'))


def _read(path):
    from result_storage import read_json
    return read_json(path)


def validate_task_result(task, sources, versions):
    """Only a complete, matching real result enters the batch summary."""
    from models import model_from_dict
    from runtime_identity import check_identity, digest, normalize
    from unitary_circuit import resolve_kernel_profile
    r=task['request'];args=r['arguments'];folder=Path(r['outdir'])
    reported=_read(folder/'resolved_request.json')
    if reported['cases']!=r['cases'] or reported['backend']!=r['backend'] or reported['resources']!=r['resources']:
        raise ValueError('Task resolved request differs from batch plan')
    report=_read(folder/'mps_probability_probe.json')
    if report.get('execution_status')!='passed' or len(report.get('rows',[]))!=1:
        raise ValueError('Task did not complete exactly one planned MPS setting')
    summary=report['rows'][0]
    # Summary holds a path to authoritative evidence rather than a second log.
    label=f"{summary['case']['id']}_bond{summary['bond']}_trunc{summary['truncation_threshold']:g}.json"
    result_path=folder/label
    if result_path.resolve().parent!=folder.resolve():raise ValueError('Task result path escaped its run')
    row=_read(result_path)
    reference=summary.get('metadata_log_references',{}).get('MPS_log_data')
    if 'MPS_log_data' in row.get('metadata',{}) and reference is None:
        raise ValueError('Missing authoritative result hash')
    if reference is not None:
        from result_storage import content_digest
        if reference['path']!=label or content_digest(result_path)['sha256']!=reference['file_sha256']:
            raise ValueError('Authoritative result hash mismatch')
    if row.get('status')!='complete' or row.get('execution_status')!='passed':raise ValueError('Incomplete task evidence')
    import math
    if len(row.get('probabilities',[]))!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row['probabilities']):
        raise ValueError('Invalid four-state probabilities')
    for field in ('probabilities','reference','rare_accuracy_status','strict_status','work_accuracy_status',
                  'raw_probability_sum','normalization_residual','numerical_validity_status',
                  'unresolved_numerical_issues','scientific_certification_passed','work_bit_probabilities'):
        if summary.get(field)!=row.get(field):raise ValueError('Summary/raw evidence mismatch: '+field)
    identity=row['identity'];check_identity(identity)
    expected_config=normalize(model_from_dict(r['cases'][0]['model_document']))
    if identity['sources']!=sources or identity['versions']!=versions:raise ValueError('Source/environment identity mismatch')
    if identity['phase']!='mps_probe' or identity['config']!=expected_config or identity['extra']['steps']!=args['steps']:
        raise ValueError('Task model/steps/phase identity mismatch')
    if identity['actual_profile']!=normalize(asdict(resolve_kernel_profile(kernel_profile=args['profile']))):
        raise ValueError('Task profile identity mismatch')
    expected_options=r['backend']['per_case_options'][0]['first_setting']
    if row['backend']['requested']!=expected_options or row['blas_threads']!=args['blas_threads']:
        raise ValueError('Task backend identity mismatch')
    if not row.get('original_A_fingerprint') or not row.get('compiled_A_fingerprint'):
        raise ValueError('Missing circuit fingerprints')
    if identity['extra']['original_A_fingerprint']!=row['original_A_fingerprint']:
        raise ValueError('Original circuit identity mismatch')
    for field in ('compiled_A_fingerprint','diagnostic_fingerprint'):
        if not row.get(field) or identity['extra'].get(field)!=row[field]:
            raise ValueError('Compiled/diagnostic circuit identity mismatch: '+field)
    if identity['backend']!=row['backend'] or identity['compile']!={'basis_gates':['u','cx'],
        'optimization_level':0,'seed_transpiler':917,'num_processes':1}:
        raise ValueError('Compilation/backend identity mismatch')
    # No normalization, clipping or fabricated zero; keep all accuracy states.
    return {key:row.get(key) for key in ('probabilities','reference','absolute_errors','p_D_relative_error',
        'rare_accuracy_status','strict_status','work_accuracy_status','work_any_nonzero_union_upper_bound',
        'work_bit_probabilities','raw_probability_sum','normalization_residual','numerical_validity_status',
        'unresolved_numerical_issues','scientific_certification_passed','certification_scope',
        'statistical_accuracy_status','physical_discretization_status','diagnostic_fingerprint',
        'original_A_fingerprint','compiled_A_fingerprint','bond','truncation_threshold')} | {
        'result_path':str(result_path.relative_to(folder.parent.parent)),
        'identity_hash':identity['identity_hash'],'statistical_estimation_status':'not_performed'}


def _execute(plan):
    from runtime_identity import prepare_run_directory, write_json_new, source_manifest, digest
    from supervised import run_limited, publish_progress
    out=prepare_run_directory(plan['outdir'],reserved_subdirectories=('_batch_supervision',))
    write_json_new(out/'batch_request.json',plan)
    sources=plan['identity']['sources'];versions=plan['identity']['versions']
    def launch(task):
        args=task['request']['arguments'];folder=prepare_run_directory(task['request']['outdir'])
        token=__import__('uuid').uuid4().hex
        command=[sys.executable,'-B',str(ROOT/'main.py'),'--phase','mps_probe']
        flags={'config':'--config','suite':'--suite','case_id':'--case-id','steps':'--steps','profile':'--kernel-profile','bond':'--mps-max-bond',
            'mps_trunc':'--mps-trunc','mps_lapack':'--mps-lapack','counts_check':'--counts-check',
            'shots':'--shots','seed':'--seed','cpu_budget':'--cpu-budget','aer_threads':'--aer-threads',
            'mps_threads':'--mps-threads','blas_threads':'--blas-threads','memory_mib':'--run-memory-mib','outdir':'--outdir'}
        for name,flag in flags.items():
            v=args[name]
            if v is not None:command += [flag,str(int(v)) if isinstance(v,bool) else str(v)]
        command += ['--internal-worker',token]
        row={'id':task['id'],'task_request_hash':digest(task['request']),'started_unix':time.time(),
            'execution_status':'failed','accuracy_status':'not_evaluated','result':None}
        try:
            result=run_limited(command,cwd=ROOT,output_dir=folder/'_supervision',timeout_sec=None,
                memory_bytes=args['memory_mib']*1024**2,threads=args['aer_threads'],blas_threads=args['blas_threads'],
                env_extra={'VRQO_MAIN_WORKER_TOKEN':token,'VRQO_CMD_REQUEST_JSON':'{}'},
                echo_output=False,heartbeat_sec=30,heartbeat_tag='batch '+task['id'])
            write_json_new(folder/'supervision_result.json',result)
            row['supervision']=str((folder/'supervision_result.json').relative_to(out))
            row['execution_status']=result['status']
            row['all_processes_reaped']=result['all_processes_reaped']
            row['process_tree_resources']={key:result.get(key) for key in
                ('tree_cpu_seconds','cpu_average_cores','elapsed_seconds','cpu_elapsed_seconds',
                 'peak_tree_rss_bytes','memory_limit_bytes','cpu_accounting_source','cpu_accounting_coverage')}
            if result['status']=='passed':
                row['result']=validate_task_result(task,sources,versions)
                row['accuracy_status']='recorded_separately'
        except Exception as exc:
            import traceback
            row.update(execution_status='failed',error=repr(exc),traceback=traceback.format_exc(),result=None)
        row['finished_unix']=time.time()
        write_json_new(folder/'batch_task_result.json',row)
        return row
    rows=[];start=time.time();overlap=[];max_active=0;previous_pids=None
    # Outer supervisor owns every descendant; Ctrl+C or coordinator failure
    # terminates/reaps all tasks, including those currently in Aer native code.
    with ThreadPoolExecutor(max_workers=plan['resources']['batch_workers']) as executor:
        futures={executor.submit(launch,task):task['id'] for task in plan['tasks']}
        pending=set(futures)
        while pending:
            active=[]
            for child in psutil.Process().children(recursive=True):
                try:
                    command=child.cmdline()
                    if '--internal-worker' in command and 'mps_probe' in command:
                        active.append({'pid':child.pid,'create_time':child.create_time()})
                except psutil.Error:pass
            pids=sorted(r['pid'] for r in active)
            max_active=max(max_active,len(pids))
            if pids!=previous_pids:
                overlap.append({'wall_unix':time.time(),'live_transport_workers':active})
                previous_pids=pids
            done,pending=wait(pending,timeout=.05,return_when=FIRST_COMPLETED)
            for future in done:
                rows.append(future.result())
                publish_progress('batch_task_complete',task_id=rows[-1]['id'],completed=len(rows),total=len(futures))
    order={task['id']:i for i,task in enumerate(plan['tasks'])}
    rows.sort(key=lambda row:order[row['id']])
    changed=source_manifest()!=sources
    if changed:
        for row in rows:row.update(execution_status='failed',error='Source changed while batch was running',result=None)
    success=not changed and len(rows)==len(plan['tasks']) and all(r['execution_status']=='passed' and r['result'] is not None for r in rows)
    report={'schema_version':1,'identity_hash':plan['identity_hash'],'execution_status':'passed' if success else 'failed',
        'planned_task_count':len(plan['tasks']),'completed_task_count':sum(r['execution_status']=='passed' for r in rows),
        'tasks':rows,'resources':plan['resources'],'elapsed_seconds':time.time()-start,
        'max_observed_live_transport_workers':max_active,'process_overlap_samples':overlap,
        'statistical_estimation_status':'not_performed','resume_supported':False,
        'resource_measurement_note':'Per-task process tree CPU times are measured; aggregate coordinator and task totals are in batch_supervision_result.json. Requested threads are budgets, not measured cores.',
        'summary_policy':'Only complete identity-matched result values; failed tasks retain null result, never zero.'}
    write_json_new(out/'batch_summary.json',report)
    print(json.dumps({'batch_summary':str(out/'batch_summary.json'),'execution_status':report['execution_status']}),flush=True)
    return 0 if success else 3


def main(argv=None):
    from runtime_identity import prepare_run_directory, write_json_new, new_run_id
    from supervised import run_limited
    from run_request import resolve_run_request
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--phase',choices=['mps_batch'],default='mps_batch')
    for name in ('suite','steps','bond','mps_trunc','mps_lapack','cpu_budget','aer_threads','mps_threads',
                 'blas_threads','memory_mib','batch_memory_mib','batch_workers','counts_check','shots','seed','outdir'):
        alias={'memory_mib':'run-memory-mib','bond':'mps-max-bond'}.get(name,name.replace('_','-'))
        flags=['--'+alias]
        if name=='memory_mib':flags.append('--memory-mib')
        parser.add_argument(*flags,dest=name,default=None)
    parser.add_argument('--kernel-profile',dest='profile')
    parser.add_argument('--dry-run',action='store_true',default=None)
    parser.add_argument('--internal-batch-worker',default=None,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if os.environ.get('VRQO_DEADLINE_MONOTONIC'):raise ValueError('Inherited wall deadline forbidden')
    cli={k:v for k,v in vars(args).items() if k not in ('phase','internal_batch_worker') and v is not None}
    plan=resolve_run_request({'cli':dict(task='mps_batch',**cli),'cmd':json.loads(os.environ.get('VRQO_CMD_REQUEST_JSON','{}'))},project_root=ROOT)
    if plan['dry_run']:
        print(json.dumps({'status':'validated_only','request':plan,'simulation_executed':False}),flush=True);return 0
    if args.internal_batch_worker:
        if args.internal_batch_worker!=os.environ.get('VRQO_BATCH_WORKER_TOKEN'):raise ValueError('Invalid batch worker token')
        return _execute(plan)
    out=prepare_run_directory(plan['outdir']);token=new_run_id()
    command=[sys.executable,'-B',str(ROOT/'main.py'),*(sys.argv[1:] if argv is None else argv),
             '--outdir',str(out),'--internal-batch-worker',token]
    result=run_limited(command,cwd=ROOT,output_dir=out/'_batch_supervision',timeout_sec=None,
        memory_bytes=plan['resources']['batch_memory_mib']*1024**2,threads=1,blas_threads=1,
        env_extra={'VRQO_BATCH_WORKER_TOKEN':token},echo_output=True,heartbeat_sec=30,heartbeat_tag='MPS batch aggregate')
    write_json_new(out/'batch_supervision_result.json',result)
    return 0 if result['status']=='passed' else 3
