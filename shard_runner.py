"""Run one immutable shard plan; resume only verified complete v2 shards."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import subprocess
import time
import traceback
import psutil
import math
from result_storage import finalize_log, open_result, read_json, result_exists
from runtime_identity import (validate_output_path, prepare_run_directory, assert_new_file,
    write_json_new, new_run_id, digest, validate_completed_result,validate_result_request,validate_cache_metadata)


def _parse_extra(text):return shlex.split(text, posix=(os.name!='nt')) if text else []


def build_plan(args):
    if args.num_shards<=0 or args.max_parallel<=0 or args.seed0<0 or args.steps<=0:
        raise ValueError('Positive steps/shards/workers and nonnegative seed required')
    total=args.histories_per_shard*args.num_shards if args.phase=='mc' else (args.total_shots or args.shots_per_shard*args.num_shards)
    if total<=0 or args.shot_chunk<0:raise ValueError('Invalid total samples or chunk')
    if args.phase=='test' and args.shot_chunk:
        chunks,remainder=divmod(total,args.shot_chunk);base,extra=divmod(chunks,args.num_shards)
        sizes=[(base+(i<extra))*args.shot_chunk for i in range(args.num_shards)];sizes[-1]+=remainder
    else:
        base,extra=divmod(total,args.num_shards);sizes=[base+(i<extra) for i in range(args.num_shards)]
    plan=[];offset=0
    for sid,size in enumerate(sizes):
        if size:plan.append({'sid':sid,'shots':size,'global_offset':offset})
        offset+=size
    return total,plan


def _read(path):return read_json(path)


def coordinate_shards(items, command_for, *, out, cwd, max_parallel, memory_bytes,
                      threads=1, blas_threads=1, heartbeat_sec=60., resource_sample_sec=1.,
                      poll_sec=.1, debug=False, total_samples=0, completed_shards=0, completed_samples=0,
                      inherited_memory_guard=False):
    """One writer; abort unsafe startup, isolate confirmed worker failures; no retry."""
    import supervised
    if len({i['sid'] for i in items}) != len(items):raise ValueError('Duplicate shard id')
    if max_parallel<=0 or any(not math.isfinite(v) or v<=0 for v in (resource_sample_sec,poll_sec)):
        raise ValueError('Positive finite scheduler intervals and parallelism required')
    if not math.isfinite(heartbeat_sec) or heartbeat_sec<0:raise ValueError('Finite nonnegative heartbeat interval required')
    out=validate_output_path(out);logs=out/'launch_logs';logs.mkdir(parents=True,exist_ok=True)
    rows=[dict(sid=i['sid'],state='pending',pid=None,shots_completed=0,history=['pending']) for i in items]
    active={};handles=[];failures=[];job=None;started=time.monotonic();peak_running=0
    last_sample=last_heartbeat=last_launch=started;resources={};science_cpu=None;previous_cpu={};previous_stamp=None
    started_count=0;all_reaped=False;current=None
    snapshot_interval=heartbeat_sec or 60.
    observation_interval=max(resource_sample_sec,snapshot_interval)
    next_snapshot=started;snapshot_status='unavailable';snapshot_write_failures=0
    def save(*,force=False):
        nonlocal next_snapshot,snapshot_status,snapshot_write_failures
        # This is an observation snapshot, never startup authority or a result.
        # Keep the last valid file on failure, and avoid retrying for every shard.
        if not force and snapshot_status=='unavailable' and time.monotonic()<next_snapshot:return
        published=supervised._display_record(out/'batch_state.json',{'states':rows,'failures':failures,
            'updated_monotonic':time.monotonic(),'updated_unix':time.time()})
        snapshot_status='current' if published else 'unavailable'
        if not published:snapshot_write_failures+=1
        next_snapshot=time.monotonic()+snapshot_interval
    def transition(row,state):
        row['state']=state;row['history'].append(state);save()
    def fail(row,reason):
        sid=None if row is None else row['sid'];path=logs/('batch_error.log' if sid is None else f'shard_{sid:03d}.log')
        failure={'shard_id':sid,'reason':str(reason),'error_log':str(path)};failures.append(failure)
        if row is not None:row.update(state='failed',error=str(reason));row['history'].append('failed')
        # Console evidence precedes disk I/O, so disk errors cannot conceal failure.
        print(f'[ERROR] shard={sid} {reason} | log={path}',flush=True)
        try:
            with path.open('a',encoding='utf8') as stream:
                stream.write(str(reason)+'\n')
                if sys.exc_info()[0] is not None:stream.write(traceback.format_exc()+'\n')
        except OSError as exc:print(f'[ERROR] cannot write error log: {exc}',flush=True)
    def summary(tag='HB'):
        elapsed=int(time.monotonic()-started);counts={s:sum(r['state']==s for r in rows) for s in ('pending','starting','started','running','complete','failed')}
        cpu=resources.get('batch_cpu_interval_average_cores');ram=resources.get('batch_rss_mib')
        cpu_text='unavailable' if cpu is None else f'{cpu:.2f} cores'
        sci_text='unavailable' if science_cpu is None else f'{science_cpu:.2f} cores'
        ram_text='unavailable' if ram is None else f'{ram:.0f} MiB'
        inflight=any(r.get('stage')=='aer_submit' and r['state']=='running' for r in rows)
        progress='batch computing, completed count awaiting batch return' if inflight else 'observed completed samples'
        snapshot_note=(f' | state_snapshot=unavailable failed_updates={snapshot_write_failures}'
                       if snapshot_status=='unavailable' else '')
        print(f'[{tag}] elapsed={elapsed//3600:02d}:{elapsed//60%60:02d}:{elapsed%60:02d} '
              f'started={started_count+completed_shards}/{len(rows)+completed_shards} '
              f'running={counts["running"]+counts["started"]} done={counts["complete"]+completed_shards} '
              f'failed={counts["failed"]} pending={counts["pending"]} starting={counts["starting"]} | '
              f'shots={completed_samples+sum(r["shots_completed"] for r in rows)}/{total_samples}, {progress} | '
              f'CPU={cpu_text} science_CPU={sci_text} RAM={ram_text}{snapshot_note}',flush=True)
        unfinished=[r['sid'] for r in rows if r['state'] in ('pending','starting','failed')]
        if unfinished:print(f'[{tag}] pending/starting/failed shard ids: '+','.join(f'{sid:03d}' for sid in unfinished),flush=True)
    try:
        save()
        # The outer supervisor owns the aggregate memory cap. This nested Job
        # only gives the coordinator a handle for its workers and descendants.
        if os.name=='nt':job=supervised._WindowsJob(None if inherited_memory_guard else memory_bytes)
        index=0
        while index<len(items) or active:
            # Poll every existing process between launches: a late-submitted failure
            # never waits for an earlier multi-day calculation to finish.
            for sid,(proc,row,item,ack,progress,launched) in list(active.items()):
                current=row;rc=proc.poll()
                if row['state']=='starting' and ack.exists():
                    confirmation=_read(ack)
                    if confirmation.get('pid')!=proc.pid:raise ValueError(f'shard {sid} startup PID mismatch')
                    row['confirmed_started']=True;started_count+=1;transition(row,'started')
                    transition(row,'running')
                if rc is not None:
                    row['returncode']=rc
                    if rc or not row.get('confirmed_started'):
                        error_log=logs/f'shard_{sid:03d}.log'
                        try:
                            tail=error_log.read_text(encoding='utf8',errors='replace')[-600:].strip().splitlines()
                            detail=tail[-1] if tail else 'no worker error text'
                        except OSError as exc:detail=f'worker log unavailable: {exc}'
                        code=f'{rc} (0x{rc & 0xffffffff:08X})' if os.name=='nt' else str(rc)
                        native=' STATUS_ACCESS_VIOLATION' if os.name=='nt' and rc & 0xffffffff==0xC0000005 else ''
                        reason=f'shard {sid} worker exited rc={code}{native}, confirmed_started={row.get("confirmed_started",False)}: {detail}'
                        if not row.get('confirmed_started'):raise RuntimeError(reason)
                        if rc in (-2,130,-1073741510,3221225786):raise KeyboardInterrupt(reason)
                        fail(row,reason);save()
                    else:
                        row['shots_completed']=item.get('shots',0);transition(row,'complete')
                    del active[sid]
                elif row['state']=='starting' and time.monotonic()-launched>120:
                    raise TimeoutError(f'shard {sid} worker did not confirm startup within 120s')
            if index<len(items) and len(active)<max_parallel:
                item=items[index];row=rows[index];current=row;sid=item['sid'];transition(row,'starting')
                token=new_run_id();ack=logs/f'shard_{sid:03d}.started.json';progress=logs/f'shard_{sid:03d}.progress.json'
                env=os.environ.copy()
                for key in ('VRQO_CMD_REQUEST_JSON','VRQO_PARENT_STATUS_FILE','VRQO_SHARD_WORKER_TOKEN'):
                    env.pop(key,None)
                env.update(VRQO_MAIN_WORKER_TOKEN=token,VRQO_WORKER_STARTED_FILE=str(ack),VRQO_PROGRESS_FILE=str(progress),
                           PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
                for key in ('OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(threads)
                for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VRQO_BLAS_THREADS'):env[key]=str(blas_threads)
                command=command_for(item)
                # Only main.py consumes the authenticated token; test children use env.
                if any(Path(str(arg)).name=='main.py' for arg in command):command=[*command,'--internal-worker',token]
                log=assert_new_file(logs/f'shard_{sid:03d}.log')
                with log.open('x',encoding='utf8') as stream:
                    proc=subprocess.Popen(command,cwd=str(cwd),env=env,stdout=stream,stderr=stream,
                        creationflags=0x00000004 if os.name=='nt' else 0,start_new_session=os.name!='nt')
                handles.append(proc);row['pid']=proc.pid
                row['create_time']=psutil.Process(proc.pid).create_time()
                active[sid]=(proc,row,item,ack,progress,time.monotonic())
                if job:job.assign(proc);job.resume(proc)
                save();index+=1;peak_running=max(peak_running,len(active))
                continue
            now=time.monotonic()
            # Only the outer supervisor samples safety resources. Read display
            # observations once per heartbeat instead of opening 50 files/second.
            if now-last_sample>=observation_interval:
                current=None
                for proc,row,item,ack,progress,launched in active.values():
                    observed=supervised._latest_progress(progress)
                    row['stage']=observed.get('stage','unknown')
                    if isinstance(observed.get('shots_completed'),int):row['shots_completed']=observed['shots_completed']
                resource_file=os.environ.get('VRQO_RESOURCE_FILE')
                if resource_file:
                    try:resources=_read(resource_file)
                    except (OSError,ValueError):resources={}
                stamp=resources.get('updated_monotonic');observed_cpu={}
                if not isinstance(stamp,(int,float)) or now-stamp>max(3.,3*observation_interval):
                    resources={};stamp=None;science_cpu=None;previous_stamp=None;previous_cpu={}
                identities={(r['pid'],r.get('create_time')) for r in rows if r.get('confirmed_started')}
                for p in resources.get('processes',[]):
                    if (p['pid'],p['create_time']) in identities:observed_cpu[p['pid']]=p['cpu_seconds']
                if stamp is not None and previous_stamp is not None and stamp>previous_stamp:
                    common=observed_cpu.keys() & previous_cpu.keys()
                    science_cpu=sum(max(0,observed_cpu[k]-previous_cpu[k]) for k in common)/(stamp-previous_stamp) if common else None
                if stamp is not None and stamp!=previous_stamp:previous_cpu=observed_cpu;previous_stamp=stamp
                last_sample=now
            if now>=next_snapshot:save()
            if heartbeat_sec and now-last_heartbeat>=heartbeat_sec:summary();last_heartbeat=now
            if started_count and now-last_launch>=5 and started_count<len(rows):summary('START');last_launch=now
            time.sleep(poll_sec)
    except (Exception,KeyboardInterrupt) as exc:
        fail(current,repr(exc))
    finally:
        # A single owned Job includes grandchildren, even if a worker already exited.
        try:
            if job:
                job.terminate()
                deadline=time.monotonic()+10
                while job.active_processes() and time.monotonic()<deadline:time.sleep(.05)
                all_reaped=job.active_processes()==0
            for proc in handles:
                if os.name!='nt':
                    import signal
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                if proc.poll() is None:proc.kill()
                proc.wait(timeout=10)
                if os.name=='nt' and not supervised._windows_handle_signaled(proc._handle,10_000):
                    raise subprocess.TimeoutExpired(proc.args,10)
            if job is None:all_reaped=all(p.poll() is not None for p in handles)
        except (Exception,KeyboardInterrupt) as exc:fail(None,'cleanup: '+repr(exc));all_reaped=False
        finally:
            if job:job.close()
        for row in rows:
            if row['state'] in ('starting','started','running'):
                row.update(state='failed',error='batch aborted; owned worker terminated');row['history'].append('failed')
        save(force=True)
    summary('END')
    return {'states':rows,'failures':failures,'all_processes_reaped':all_reaped,
            'peak_running':peak_running,'started_shards':started_count,'elapsed_seconds':time.monotonic()-started,
            'snapshot_status':snapshot_status,'snapshot_write_failures':snapshot_write_failures,
            'snapshot_interval_sec':snapshot_interval,'observation_interval_sec':observation_interval,
            'memory_guard':'inherited_batch_job' if inherited_memory_guard else 'coordinator_job'}


def validate_batch_cache(path,request,completed):
    from qiskit import qpy
    path=validate_output_path(path)
    with path.open('rb') as f:circuits=qpy.load(f)
    if len(circuits)!=1:raise ValueError('Cache must contain exactly one circuit')
    meta=_read(str(path)+'.meta.json')
    fingerprint=validate_cache_metadata(meta,meta.get('identity'),circuits[0])
    validate_result_request({'identity':meta['identity'],'steps':request['steps']},request)
    for result in completed:
        if result['identity'].get('extra',{}).get('circuit_fingerprint')!=fingerprint:
            raise ValueError('Completed shard did not run the retained compiled circuit')
    return fingerprint


def _run(command, log, cwd, *, heartbeat_sec=60., memory_bytes=None, resource_sample_sec=1., poll_sec=.1, debug=False):
    from supervised import run_limited
    log=assert_new_file(log);log.parent.mkdir(parents=True,exist_ok=True)
    supervision=log.parent/(log.stem+'__supervision_'+new_run_id()[:12])
    with log.open('x',encoding='utf-8') as f:
        f.write('COMMAND '+json.dumps(command)+'\nLIVE_LOGS '+str(supervision)+'\n');f.flush()
        result=run_limited(command,cwd=cwd,output_dir=supervision,timeout_sec=None,
            memory_bytes=memory_bytes,
            threads=int(os.environ.get('OMP_NUM_THREADS','1')),echo_output=debug,
            heartbeat_sec=heartbeat_sec,heartbeat_tag='compile',resource_sample_sec=resource_sample_sec,poll_sec=poll_sec,debug=debug)
        f.write(f'RESULT {result["status"]} {supervision / "process_result.json"}\n')
    finalize_log(log)
    return 0 if result['status']=='passed' else (result.get('returncode') or 3)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['test','mc','iqae'],required=True)
    ap.add_argument('--steps',type=int,default=5)
    ap.add_argument('--num-shards',type=int,default=1)
    ap.add_argument('--max-parallel',type=int,default=1)
    ap.add_argument('--seed0',type=int,default=1234)
    ap.add_argument('--outroot',required=True)
    ap.add_argument('--shots-per-shard',type=int,default=200)
    ap.add_argument('--total-shots',type=int,default=0)
    ap.add_argument('--shot-chunk',type=int,default=0)
    ap.add_argument('--histories-per-shard',type=int,default=200000)
    ap.add_argument('--extra',default='')
    ap.add_argument('--extra-json',default='',help=argparse.SUPPRESS)
    ap.add_argument('--compile-cache',action='store_true')
    ap.add_argument('--no-compile-cache',action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--internal-worker',default='',help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.extra and args.extra_json:raise ValueError('Use only one extra argument encoding')
    extra=json.loads(args.extra_json) if args.extra_json else _parse_extra(args.extra)
    if not isinstance(extra,list) or not all(isinstance(t,str) for t in extra):raise ValueError('extra-json must contain a string list')
    if args.phase=='iqae':raise ValueError('IQAE repeats are separate estimator runs; use main.py and retain per-power records, not shard mean merging')
    from main import build_argparser, invocation_identity, _iqae_backend_options
    root=Path(__file__).resolve().parent;out=validate_output_path(args.outroot)
    expected=os.environ.get('VRQO_SHARD_WORKER_TOKEN')
    if args.internal_worker and args.internal_worker!=expected:
        raise ValueError('Invalid supervised shard-worker token')
    if not args.dry_run and not args.internal_worker:
        from supervised import run_limited
        token=new_run_id()
        if args.resume:
            if not result_exists(out/'plan_meta.json'):raise ValueError('Resume requires a complete v2 plan')
        else:
            prepare_run_directory(out)
        control=out/'_runner_supervision'/token[:12]
        control.parent.mkdir(parents=True,exist_ok=True)
        lock=control.parent/'active.lock'
        with lock.open('x',encoding='utf8') as f:f.write(str(os.getpid()))
        try:
            forwarded=build_argparser().parse_args(['--phase',args.phase,*extra])
            result=run_limited([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],
                '--internal-worker',token],cwd=root,output_dir=control,timeout_sec=None,
                memory_bytes=None if forwarded.run_memory_mib is None else forwarded.run_memory_mib*1024**2,
                threads=forwarded.aer_threads,blas_threads=forwarded.blas_threads,echo_output=True,
                heartbeat_sec=0,heartbeat_tag='shard runner',resource_sample_sec=forwarded.resource_sample_sec,
                resource_publish_sec=forwarded.aer_heartbeat_sec or 60.,
                poll_sec=forwarded.poll_sec,debug=forwarded.debug_runtime,
                env_extra={'VRQO_SHARD_WORKER_TOKEN':token})
        finally:lock.unlink()
        # Resume attempts are immutable; retain every supervision outcome.
        report=out/'runner_supervision_result.json'
        if result_exists(report):report=out/('runner_supervision_result_'+token[:12]+'.json')
        write_json_new(report,{**result,'process_result_path':str(control/'process_result.json')})
        if forwarded.debug_runtime:print(json.dumps({'runner_supervision':str(report),**result},ensure_ascii=False),flush=True)
        elif result['status']!='passed':print(f'[ERROR] batch {result["status"]}; log={control / "stderr.log"}',flush=True)
        return 0 if result['status']=='passed' else 3
    forbidden={'--outdir','--phase','--steps','--shots','--histories','--seed','--shard-id','--num-shards',
        '--root-seed','--total-samples','--global-shot-offset','--run-id','--request-hash','--compiled-qpy',
        '--save-compiled-qpy','--compile-only','--internal-worker','--dry-run'}
    if any(token.split('=')[0] in forbidden for token in extra):raise ValueError('Runner-managed option cannot occur in --extra')
    effective=build_argparser().parse_args(['--phase',args.phase,'--steps',str(args.steps),*extra])
    if any(not math.isfinite(v) or v<=0 for v in (effective.resource_sample_sec,effective.poll_sec)):
        raise ValueError('Positive finite resource and polling intervals required')
    if not math.isfinite(effective.aer_heartbeat_sec) or effective.aer_heartbeat_sec<0:
        raise ValueError('Finite nonnegative heartbeat interval required')
    if args.shot_chunk and effective.aer_shot_chunk!=args.shot_chunk:
        if any(token.split('=')[0]=='--aer-shot-chunk' for token in extra):raise ValueError('Runner chunk conflicts with backend sampling chunk')
        extra+=['--aer-shot-chunk',str(args.shot_chunk)];effective.aer_shot_chunk=args.shot_chunk
    _iqae_backend_options(effective)
    total,plan=build_plan(args)
    request={'invocation':invocation_identity(effective),'phase':args.phase,'steps':args.steps,
        'root_seed':args.seed0,'total_samples':total,'shard_plan':plan,'sampling_chunk':effective.aer_shot_chunk,
        'compile_cache':args.phase=='test' and not args.no_compile_cache}
    if args.phase=='test':
        request['backend_requested']=_iqae_backend_options(effective)
        request['compile_options']={'optimization_level':0,'num_processes':1,'seed_transpiler':effective.transpiler_seed,
            'basis_gates':['rz','sx','x','cx'] if effective.aer_safe_basis else None,'skip_transpile':bool(effective.skip_transpile)}
    request_hash=digest(request);old_plan=out/'plan_meta.json'
    if args.dry_run:
        from main import _load_config
        from config import validate_quantum_config
        if args.phase=='test':validate_quantum_config(_load_config(effective),args.steps)
        if args.resume:
            if not result_exists(old_plan):raise ValueError('Resume requires a complete v2 plan')
            if _read(old_plan).get('request_hash')!=request_hash:raise ValueError('Resume configuration/source/version/plan mismatch')
        if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError(f'Output directory occupied: {out}')
        print(f'[CHECK] phase={args.phase} steps={args.steps} shards={len(plan)} parallel={args.max_parallel} '
              f'shots={total} chunk={effective.aer_shot_chunk} seed={args.seed0}; validated only, no scientific worker',flush=True)
        return 0
    if args.resume:
        if not result_exists(old_plan):raise ValueError('Resume requires a complete v2 plan')
        recorded=_read(old_plan)
        if recorded.get('request_hash')!=request_hash:raise ValueError('Resume configuration/source/version/plan mismatch')
        run_id=recorded['run_id']
    else:
        prepare_run_directory(out,
            reserved_subdirectories=('_runner_supervision',) if args.internal_worker else ())
        run_id=new_run_id()
        recorded={'schema_version':2,'run_id':run_id,'request_hash':request_hash,'request':request,
            'requested_shards':args.num_shards,'active_planned_shards':len(plan),
            'empty_planned_shards':args.num_shards-len(plan),'total_shots':total,
            'shot_chunk':effective.aer_shot_chunk,'canonical_chunk_mode':bool(args.shot_chunk),'shard_plan':plan}
        write_json_new(old_plan,recorded)
    pending=[];completed=[]
    for item in plan:
        sd=out/'shards'/f"shard_{item['sid']:03d}";result=sd/'results.json'
        sample={'sample_start':item['global_offset'],'sample_stop':item['global_offset']+item['shots'],
            'total_samples':total,'num_shards':len(plan),'shard_id':item['sid'],'root_seed':args.seed0,
            'chunk':effective.aer_shot_chunk if args.phase=='test' else None,
            'seed_strategy':'splitmix64(root_seed, abs_shot_offset, chunk_shots)' if args.phase=='test' else 'root_seed + shard_id'}
        if result_exists(result):
            if not args.resume:raise FileExistsError('Existing shard requires explicit --resume')
            js=_read(result);validate_completed_result(js,expected_sample=sample,request_hash=request_hash)
            validate_result_request(js,request)
            if js['run_id']!=run_id:raise ValueError('Resume run_id mismatch')
            completed.append(js)
        else:
            if sd.exists() and any(sd.iterdir()):raise ValueError(f'Partial shard directory is not resumable: {sd}; retain it and start a new run')
            pending.append(item)
    print(f'[RUN] phase={args.phase} steps={args.steps} shards={len(plan)} parallel={args.max_parallel} '
          f'shots={total} chunk={effective.aer_shot_chunk} seed={args.seed0} bond={effective.mps_max_bond} '
          f'threads={effective.aer_threads}/{effective.mps_threads}/{effective.blas_threads} '
          f'heartbeat={effective.aer_heartbeat_sec}s resource_sample={effective.resource_sample_sec}s',flush=True)
    cached=out/'compiled'/'test.qpy'
    if request['compile_cache']:
        if cached.exists():
            validate_batch_cache(cached,request,completed)
        elif pending:
            cmd=[sys.executable,'-B',str(root/'main.py'),'--phase','test','--steps',str(args.steps),'--shots','1',
                '--compile-only','--save-compiled-qpy',str(cached),'--outdir',str(out/'compile_run'),*extra]
            rc=_run(cmd,out/'launch_logs'/'compile.log',root,heartbeat_sec=effective.aer_heartbeat_sec,
                memory_bytes=None if effective.run_memory_mib is None else effective.run_memory_mib*1024**2,
                resource_sample_sec=effective.resource_sample_sec,poll_sec=effective.poll_sec,debug=effective.debug_runtime)
            if rc:
                print(f'[ERROR] compilation failed rc={rc}; log={out / "launch_logs" / "compile.log"}',flush=True)
                return rc
    def launch(item):
        sid=item['sid'];seed=args.seed0 if args.phase=='test' else args.seed0+sid
        cmd=[sys.executable,'-B',str(root/'main.py'),'--phase',args.phase,'--steps',str(args.steps),
            '--shots' if args.phase=='test' else '--histories',str(item['shots']),'--seed',str(seed),
            '--shard-id',str(sid),'--num-shards',str(len(plan)),'--outdir',str(out),
            '--global-shot-offset',str(item['global_offset']),'--total-samples',str(total),
            '--root-seed',str(args.seed0),'--run-id',run_id,'--request-hash',request_hash,*extra]
        if request['compile_cache']:cmd+=['--compiled-qpy',str(cached)]
        return cmd
    outcome=coordinate_shards(pending,launch,out=out,cwd=root,max_parallel=args.max_parallel,
        memory_bytes=effective.run_memory_mib*1024**2 if effective.run_memory_mib else min(4*1024**3,psutil.virtual_memory().available//4),
        threads=effective.aer_threads,blas_threads=effective.blas_threads,heartbeat_sec=effective.aer_heartbeat_sec,
        resource_sample_sec=effective.resource_sample_sec,poll_sec=effective.poll_sec,debug=effective.debug_runtime,
        total_samples=total,completed_shards=len(completed),
        completed_samples=sum(p['shots'] for p in plan if p not in pending),
        inherited_memory_guard=bool(args.internal_worker and os.environ.get('VRQO_SUPERVISED')=='1'))
    write_json_new(out/f'runner_attempt_{new_run_id()}.json',{'run_id':run_id,'request_hash':request_hash,
        'resumed_shards':len(completed),'submitted_shards':len(pending),**outcome})
    if outcome['failures'] or not outcome['all_processes_reaped']:return 2
    from merge_shards import merge_test,merge_mc
    files=[str(out/'shards'/f"shard_{p['sid']:03d}"/'results.json') for p in plan]
    merged=(merge_test if args.phase=='test' else merge_mc)(files)
    destination=out/'merged'
    if result_exists(destination/'results_merged.json'):
        if _read(destination/'results_merged.json')!=merged:raise ValueError('Existing merged result differs')
    else:
        prepare_run_directory(destination);write_json_new(destination/'results_merged.json',merged)
    print(f'[RESULT] status=complete strict_merge=passed shards={len(plan)} samples={total}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
