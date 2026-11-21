"""Bounded foreground subprocesses. No simulation runs in the supervisor.

Windows uses a Job Object with a hard aggregate commit limit and kill-on-close;
all platforms additionally monitor aggregate RSS and reap process descendants.
Linux additionally uses RLIMIT_AS (a stricter per-process virtual-memory limit).
"""
from __future__ import annotations
import ctypes
import codecs
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import psutil


class _WindowsJob:
    def __init__(self, memory_bytes):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),('LimitFlags',w.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',w.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_uint64) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR]; self.kernel.CreateJobObjectW.restype=w.HANDLE
        self.kernel.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]; self.kernel.SetInformationJobObject.restype=w.BOOL
        self.kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]; self.kernel.AssignProcessToJobObject.restype=w.BOOL
        self.kernel.CloseHandle.argtypes=[w.HANDLE]; self.kernel.CloseHandle.restype=w.BOOL
        self.handle=self.kernel.CreateJobObjectW(None,None)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        self._extended_type=Extended
        data=Extended(); data.BasicLimitInformation.LimitFlags=0x2000  # KILL_ON_JOB_CLOSE
        if memory_bytes is not None:
            data.BasicLimitInformation.LimitFlags|=0x200  # JOB_MEMORY
            data.JobMemoryLimit=int(memory_bytes)
        if not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(data),ctypes.sizeof(data)):
            self.close(); raise ctypes.WinError(ctypes.get_last_error())
    def assign(self,proc):
        if not self.kernel.AssignProcessToJobObject(self.handle,int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
    def resume(self,proc):
        # CREATE_SUSPENDED prevents user code and grandchildren before assignment.
        # APIs: Microsoft Learn CreateProcess flags / OpenThread / ResumeThread.
        from ctypes import wintypes as w
        self.kernel.OpenThread.argtypes=[w.DWORD,w.BOOL,w.DWORD];self.kernel.OpenThread.restype=w.HANDLE
        self.kernel.ResumeThread.argtypes=[w.HANDLE];self.kernel.ResumeThread.restype=w.DWORD
        threads=psutil.Process(proc.pid).threads()
        if len(threads)!=1:raise RuntimeError('Expected one suspended initial thread')
        handle=self.kernel.OpenThread(0x0002,False,threads[0].id)  # THREAD_SUSPEND_RESUME
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        try:
            count=self.kernel.ResumeThread(handle)
            if count!=1:raise RuntimeError(f'Unexpected initial thread suspend count: {count}')
        finally:self.kernel.CloseHandle(handle)
    def close(self):
        if self.handle:
            if not self.kernel.CloseHandle(self.handle):raise ctypes.WinError(ctypes.get_last_error())
            self.handle=None

    def terminate(self):
        # Operate on the owned Job handle, never a potentially recycled PID.
        from ctypes import wintypes as w
        self.kernel.TerminateJobObject.argtypes=[w.HANDLE,w.UINT];self.kernel.TerminateJobObject.restype=w.BOOL
        if not self.kernel.TerminateJobObject(self.handle,1):raise ctypes.WinError(ctypes.get_last_error())

    def _accounting(self):
        from ctypes import wintypes as w
        class Accounting(ctypes.Structure):
            _fields_=[(n,ctypes.c_int64) for n in ('TotalUserTime','TotalKernelTime','ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime')]+[(n,w.DWORD) for n in ('TotalPageFaultCount','TotalProcesses','ActiveProcesses','TotalTerminatedProcesses')]
        self.kernel.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype=w.BOOL
        data=Accounting()
        if not self.kernel.QueryInformationJobObject(self.handle,1,ctypes.byref(data),ctypes.sizeof(data),None):raise ctypes.WinError(ctypes.get_last_error())
        return data

    def active_processes(self):
        return int(self._accounting().ActiveProcesses)

    def cpu_seconds(self):
        # Job lifetime totals retain CPU used by descendants that already exited.
        data=self._accounting()
        return (int(data.TotalUserTime)+int(data.TotalKernelTime))/10_000_000.

    def memory_usage(self):
        from ctypes import wintypes as w
        self.kernel.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype=w.BOOL
        data=self._extended_type()
        if not self.kernel.QueryInformationJobObject(self.handle,9,ctypes.byref(data),ctypes.sizeof(data),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'peak_job_memory_bytes':int(data.PeakJobMemoryUsed),
                'job_memory_limit_bytes':int(data.JobMemoryLimit) if data.BasicLimitInformation.LimitFlags & 0x200 else None}


def _cpu_topology():
    def count(logical):
        try:return psutil.cpu_count(logical=logical)
        except (OSError,psutil.Error):return None
    logical=count(True);physical=count(False)
    try:available=len(psutil.Process().cpu_affinity())
    except (AttributeError,OSError,psutil.Error):available=None
    return {'cpu_logical_count':logical,'cpu_physical_count':physical,
            'cpu_available_logical_count':available}


class _CPUAccounting:
    """Low-frequency measured CPU seconds; unavailable samples remain null.

    Windows Job totals cover exited children. The portable fallback retains the
    latest observed per-process times and is explicitly a sampled lower bound.
    """
    def __init__(self, started, topology=None):
        self.started=started;self.last_time=started;self.last_cpu=0.
        self.topology=_cpu_topology() if topology is None else dict(topology)
        self.observed={}

    def sample(self, now, *, job=None, processes=()):
        total=None;error=None
        source='windows_job' if job is not None else 'psutil_observed_processes'
        coverage='job_lifetime_including_exited_children' if job is not None else 'sampled_lower_bound'
        try:
            if job is not None:
                total=float(job.cpu_seconds())
            else:
                inaccessible=False
                for proc in processes:
                    try:
                        identity=(proc.pid,proc.create_time());times=proc.cpu_times()
                        self.observed[identity]=max(self.observed.get(identity,0.),times.user+times.system)
                    except psutil.NoSuchProcess:
                        continue
                    except (psutil.AccessDenied,OSError):
                        inaccessible=True
                if self.observed and not inaccessible:total=sum(self.observed.values())
                if inaccessible:error='Some process CPU times were unavailable'
            if total is not None and (not math.isfinite(total) or total < 0):
                raise ValueError('Invalid CPU time observation')
        except (OSError,psutil.Error,ValueError) as exc:
            error=repr(exc);total=None
        elapsed=now-self.started;interval=now-self.last_time
        delta=None if total is None or self.last_cpu is None else total-self.last_cpu
        cores=delta/interval if delta is not None and delta>=0 and interval>0 else None
        average=total/elapsed if total is not None and elapsed>0 else None
        available=self.topology['cpu_available_logical_count']
        row={**self.topology,'tree_cpu_seconds':total,'cpu_elapsed_seconds':elapsed,
            'cpu_interval_wall_seconds':interval,'cpu_interval_seconds':delta,
            'cpu_interval_average_cores':cores,'cpu_average_cores':average,
            'cpu_interval_available_logical_percent':100*cores/available if cores is not None and available else None,
            'cpu_available_logical_percent':100*average/available if average is not None and available else None,
            'cpu_percent_denominator':'available logical CPUs from process affinity',
            'cpu_accounting_source':source,'cpu_accounting_coverage':coverage,'cpu_accounting_error':error}
        self.last_time=now;self.last_cpu=total
        return row


def _remember_children(known, children):
    for child in children:
        try:created=child.create_time()
        except (psutil.NoSuchProcess,psutil.AccessDenied):created=None
        known.setdefault((child.pid,created),child)


def _cleanup_event(events, stage, exc):
    event={'stage':stage,'error_type':type(exc).__name__,'error':str(exc)}
    if event not in events:events.append(event)


def _probe_identity(identity, events):
    """Read-only evidence: cached is_running()/NoSuchProcess alone is insufficient."""
    pid,created=identity
    try:
        if not psutil.pid_exists(pid):return 'exited',None,None
        current=psutil.Process(pid);now=current.create_time()
    except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:
        _cleanup_event(events,'identity',exc)
        # Windows psutil can translate GetProcessTimes access errors to NSP.
        # Require an independent absence check before declaring it exited.
        try:
            if not psutil.pid_exists(pid):return 'exited',None,None
        except psutil.Error as check_exc:_cleanup_event(events,'pid_exists',check_exc)
        return 'unconfirmed',None,None
    if created is None or now is None:return 'unconfirmed',None,now
    if now!=created:return 'pid_reused',None,now
    return 'alive',current,now


def _windows_handle_signaled(handle, timeout_ms=0):
    from ctypes import wintypes as w
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];kernel.WaitForSingleObject.restype=w.DWORD
    state=kernel.WaitForSingleObject(int(handle),timeout_ms)
    if state==0:return True  # WAIT_OBJECT_0
    if state==0x102:return False  # WAIT_TIMEOUT
    raise ctypes.WinError(ctypes.get_last_error())


def _windows_process_signaled(pid):
    """Read-only native exit check; a retained process object can still have a PID.

    psutil.wait on Windows additionally polls PID disappearance, so it cannot
    distinguish that case. Call only after the saved creation time matches.
    """
    from ctypes import wintypes as w
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
    handle=kernel.OpenProcess(0x00100000,False,pid)  # SYNCHRONIZE only, no termination rights
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:return _windows_handle_signaled(handle)
    finally:kernel.CloseHandle(handle)


def _kill_tree(proc, known, *, job=None, parent_identity=None, wait_timeout=5):
    report={'parent_reaped':False,'job_active_processes':None,'job_confirmed_empty':job is None,
            'processes':[],'remaining_pids':[],'unconfirmed_pids':[],'events':[],'errors':[],
            'wait_timeout_seconds':wait_timeout,'wait_timed_out':False}
    if proc.poll() is None and parent_identity is not None:
        state,parent,_=_probe_identity(parent_identity,report['events'])
        if state=='alive':
            try:_remember_children(known,parent.children(recursive=True))
            # Discovery may itself fail under commit pressure (WinError 1455).
            # Keep that evidence, then terminate the already-owned Job handle.
            except (psutil.Error,OSError,MemoryError) as exc:_cleanup_event(report['events'],'discover_children',exc)
    if job is not None:
        try:job.terminate()
        except OSError as exc:_cleanup_event(report['errors'],'terminate_job',exc)
    if os.name!='nt':
        for identity in reversed(list(known)):
            state,current,_=_probe_identity(identity,report['events'])
            if state=='alive':
                try:current.kill()
                except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:_cleanup_event(report['events'],'kill_child',exc)
        import signal
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    # Popen retains the original Windows process handle, so this is identity-safe.
    try:
        if proc.poll() is None:proc.kill()
        proc.wait(timeout=10)
        # Some Python Windows builds poll GetExitCodeProcess without waiting for
        # the exit signal; wait() then returns the cached code during teardown.
        if os.name=='nt' and not _windows_handle_signaled(proc._handle,10_000):
            raise subprocess.TimeoutExpired(proc.args,10)
        report['parent_reaped']=True
    except (OSError,subprocess.TimeoutExpired) as exc:_cleanup_event(report['errors'],'parent_wait',exc)
    rows={identity:{'pid':identity[0],'create_time':identity[1],'state':'unconfirmed','events':[]} for identity in known}
    deadline=time.monotonic()+wait_timeout
    while True:
        pending=False
        for identity,row in rows.items():
            if row['state'] in ('exited','pid_reused'):continue
            state,current,now=_probe_identity(identity,row['events'])
            if os.name=='nt' and state=='alive':
                try:
                    if _windows_process_signaled(identity[0]):state='exited'
                except OSError as exc:
                    _cleanup_event(row['events'],'wait_child',exc)
                    state,_,now=_probe_identity(identity,row['events'])
            if os.name!='nt' and state=='alive':
                try:current.wait(timeout=0);state='exited'
                except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.TimeoutExpired) as exc:
                    _cleanup_event(row['events'],'wait_child',exc)
                    state,_,now=_probe_identity(identity,row['events'])
            row.update(state=state,observed_create_time=now)
            pending=pending or state in ('alive','unconfirmed')
        if job is not None:
            try:
                report['job_active_processes']=job.active_processes()
                report['job_confirmed_empty']=report['job_active_processes']==0
            except OSError as exc:
                _cleanup_event(report['errors'],'query_job',exc)
                report['job_confirmed_empty']=False
                break
        if not pending and report['job_confirmed_empty']:break
        remaining=deadline-time.monotonic()
        if remaining<=0:
            report['wait_timed_out']=True
            break
        time.sleep(min(.05,remaining))
    for row in rows.values():
        if row['state']=='alive':
            if report['wait_timed_out']:row['state']='wait_timeout'
            report['remaining_pids'].append(row['pid'])
        elif row['state']=='unconfirmed':report['unconfirmed_pids'].append(row['pid'])
    report['processes']=list(rows.values())
    return report


def _atomic_record(path, record):
    from runtime_identity import validate_output_path
    path=validate_output_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # A fixed-length sibling avoids doubling long result filenames on Windows.
    temporary=path.with_name('.tmp-'+uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False),encoding='utf8')
        # A Windows heartbeat reader may briefly hold the destination without
        # FILE_SHARE_DELETE. Retry only that local publication; do not abort a
        # scientific worker or silently discard a persistent permission error.
        # This bounds I/O retries, not the runtime of any computation.
        for attempt in range(9):
            try:
                os.replace(temporary,path)
                break
            except PermissionError as exc:
                if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32,33) or attempt==8:
                    raise
                time.sleep(min(.005*2**attempt,.1))
    finally:
        if temporary.exists():temporary.unlink()


_display_warnings=set()


def _warn_display(path, exc):
    key=str(path)
    if key not in _display_warnings:
        _display_warnings.add(key)
        try:
            print(f'[WARN] progress unavailable: file={path} winerror={getattr(exc,"winerror",None)} '
                  f'{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
            # In a worker this goes directly to its existing raw stderr log.
            traceback.print_exc(file=sys.stderr)
        except (OSError,UnicodeError):pass  # A failed diagnostic sink is not scientific failure.


def _display_record(path, record):
    """Display telemetry is optional; identity/result records use _atomic_record directly."""
    try:
        _atomic_record(path,record)
        key=str(path)
        if key in _display_warnings:
            _display_warnings.discard(key)
            try:print(f'[INFO] progress publication recovered: {Path(path).name}',file=sys.stderr,flush=True)
            except (OSError,UnicodeError):pass
        return True
    except Exception as exc:
        _warn_display(path,exc)
        return False


def publish_progress(stage, shots_completed=None, shots_total=None, **details):
    """Publish observations only; the independent supervisor provides liveness."""
    path=os.environ.get('VRQO_PROGRESS_FILE')
    if path:
        return _display_record(path,{'stage':stage,'shots_completed':shots_completed,'shots_total':shots_total,
            'updated_monotonic':time.monotonic(),'updated_unix':time.time(),'reporter_pid':os.getpid(),**details})


def _latest_progress(path):
    from runtime_identity import validate_output_path
    observed={};seen=set()
    try:
        for _ in range(16):
            path=validate_output_path(path)
            if path in seen:break
            seen.add(path)
            if not path.exists():break
            current=json.loads(path.read_text(encoding='utf8'))
            if not isinstance(current,dict):raise ValueError('Progress record must be an object')
            observed=current
            children=_child_records(current.get('child_status_file'))
            active=[c for c in children if c.get('status')=='running']
            if len(active)>1:
                return {'stage':'parallel_children','child_progress':[
                    {'process_result_path':c.get('process_result_path'),**_latest_progress(c['progress_file'])} for c in active],
                    'updated_monotonic':None}
            if len(active)==1:
                path=active[0]['progress_file'];continue
            if current.get('stage')=='child_finished':break
            if not current.get('child_progress_file'):break
            path=current['child_progress_file']
    except (OSError,ValueError) as exc:observed={**observed,'progress_error':repr(exc)}
    return observed


def _child_records(path):
    if not path:return []
    from runtime_identity import validate_output_path
    path=validate_output_path(path);records=[]
    for item in ([path] if path.exists() else [])+sorted(path.with_suffix('').glob('*.json')):
        try:
            value=json.loads(item.read_text(encoding='utf8'))
            if isinstance(value,dict):records.append(value)
        except (OSError,ValueError):continue
    return records


def run_limited(command, *, cwd, output_dir, timeout_sec=None, memory_bytes=None, threads=1, blas_threads=None, env_extra=None, echo_output=False, heartbeat_sec=60., heartbeat_tag='run', resource_sample_sec=1., poll_sec=.1, debug=False, resource_publish_sec=None):
    """Run once and save logs + result; status never treats timeouts as success.

    None removes this call's time limit; an explicit inherited parent deadline
    still applies. Normal calls have no automatically generated wall deadline.
    Output directories, memory limits and child cleanup apply with either setting.
    """
    if timeout_sec is not None:
        timeout_sec=float(timeout_sec)
        if not math.isfinite(timeout_sec) or timeout_sec<=0:raise ValueError('timeout must be a finite positive number or None')
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<=0:raise ValueError('threads must be a positive integer')
    blas_threads=threads if blas_threads is None else blas_threads
    if isinstance(blas_threads,bool) or not isinstance(blas_threads,int) or blas_threads<=0:raise ValueError('blas_threads must be a positive integer')
    heartbeat_sec=float(heartbeat_sec)
    if not math.isfinite(heartbeat_sec) or heartbeat_sec<0:raise ValueError('heartbeat interval must be finite and nonnegative')
    resource_sample_sec=float(resource_sample_sec);poll_sec=float(poll_sec)
    if not math.isfinite(resource_sample_sec) or resource_sample_sec<=0:raise ValueError('resource interval must be finite and positive')
    resource_publish_sec=float((heartbeat_sec or 60.) if resource_publish_sec is None else resource_publish_sec)
    if not math.isfinite(resource_publish_sec) or resource_publish_sec<=0:raise ValueError('resource publication interval must be finite and positive')
    resource_publish_sec=max(resource_publish_sec,resource_sample_sec)
    if not math.isfinite(poll_sec) or poll_sec<=0:raise ValueError('poll interval must be finite and positive')
    memory_bytes=int(min(4*1024**3,psutil.virtual_memory().available//4) if memory_bytes is None else memory_bytes)
    if memory_bytes<=0: raise ValueError('positive memory limit required')
    inherited=[]
    for raw in (os.environ.get('VRQO_DEADLINE_MONOTONIC'),(env_extra or {}).get('VRQO_DEADLINE_MONOTONIC')):
        if raw is not None and str(raw).strip():
            value=float(raw)
            if not math.isfinite(value):raise ValueError('inherited deadline must be finite')
            inherited.append(value)
    inherited_deadline=min(inherited) if inherited else None
    from runtime_identity import validate_output_path
    out=validate_output_path(output_dir); out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    if env_extra:env.update({k:str(v) for k,v in env_extra.items()})
    env.update({k:str(threads) for k in ('OMP_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    env.update({k:str(blas_threads) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    env['VRQO_BLAS_THREADS']=str(blas_threads)
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUNBUFFERED']='1'
    env['VRQO_SUPERVISED']='1';env['VRQO_OUTPUT_DIR']=str(out)
    progress_file=out/'progress.json';child_status_file=out/'child_status.json'
    parent_status_file=env.get('VRQO_PARENT_STATUS_FILE')
    parent_record=None if not parent_status_file else Path(parent_status_file).with_suffix('')/(uuid.uuid4().hex+'.json')
    env['VRQO_PROGRESS_FILE']=str(progress_file);env['VRQO_PARENT_STATUS_FILE']=str(child_status_file)
    resource_file=out/'resource.json';env['VRQO_RESOURCE_FILE']=str(resource_file)
    started=time.monotonic()
    cpu_accounting=_CPUAccounting(started)
    cpu_observation={}
    deadlines=inherited+([] if timeout_sec is None else [started+timeout_sec])
    deadline=min(deadlines) if deadlines else None
    if deadline is None:env.pop('VRQO_DEADLINE_MONOTONIC',None)
    else:env['VRQO_DEADLINE_MONOTONIC']=str(deadline)
    next_heartbeat=started+heartbeat_sec
    heartbeat_count=0
    def heartbeat(rss):
        nonlocal next_heartbeat,heartbeat_count,cpu_observation
        now=time.monotonic()
        if not heartbeat_sec or now<next_heartbeat:return
        stage=_latest_progress(progress_file)
        updated=stage.get('updated_monotonic')
        age=max(0.,now-updated) if isinstance(updated,(float,int)) and math.isfinite(updated) else None
        row={'elapsed_seconds':now-started,'pid':proc.pid,'process_alive':proc.poll() is None,
             'tree_rss_mib':None if rss is None else rss/1024**2,'peak_tree_rss_mib':peak/1024**2,
             **cpu_observation,'requested_supervisor_threads':threads,
             'stage':stage.get('stage','unknown'),'stage_age_seconds':age,
             # Age describes the observation, never diagnoses a blocked native call.
             'stage_stale':False,
             'shots_completed':stage.get('shots_completed'),'shots_total':stage.get('shots_total'),
             'reporter_pid':stage.get('reporter_pid'),'tag':heartbeat_tag,'progress_error':stage.get('progress_error'),
             'child_progress':stage.get('child_progress'),
             'stage_details':{k:v for k,v in stage.items() if k not in ('child_progress','child_status_file','child_progress_file')}}
        for child in row['child_progress'] or []:
            stamp=child.get('updated_monotonic')
            child_age=max(0.,now-stamp) if isinstance(stamp,(int,float)) else None
            child['stage_age_seconds']=child_age
            child['stage_stale']=False
        line=json.dumps(row,ensure_ascii=False,allow_nan=False)
        try:
            with (out/'supervisor_heartbeat.jsonl').open('a',encoding='utf8') as log:log.write(line+'\n');log.flush()
        except OSError as exc:_warn_display(out/'supervisor_heartbeat.jsonl',exc)
        if debug:print('SUPERVISOR '+line,flush=True)
        else:
            cores=cpu_observation.get('cpu_interval_average_cores')
            cpu='unknown' if cores is None else f'{cores:.2f} cores'
            ram='unknown' if rss is None else f'{rss/1024**2:.0f} MiB'
            shots='unknown' if row['shots_completed'] is None else str(row['shots_completed'])
            native=stage.get('stage') in ('aer_submit','aer_sampling')
            note=' | batch computing; completed shots not yet updated' if native else ''
            print(f"[HB] elapsed={now-started:.0f}s alive={row['process_alive']} stage={row['stage']} shots={shots} | CPU={cpu} RAM={ram}{note}",flush=True)
        heartbeat_count+=1;next_heartbeat=now+heartbeat_sec  # Never replay overdue ticks.
    offsets={'stdout.log':0,'stderr.log':0}
    decoders={name:codecs.getincrementaldecoder('utf8')(errors='replace') for name in offsets}
    fragments={name:'' for name in offsets}
    def echo_logs(final=False):
        if not echo_output:return
        for name,stream in (('stdout.log',sys.stdout),('stderr.log',sys.stderr)):
            path=out/name
            if not path.exists():continue
            with path.open('rb') as reader:
                reader.seek(offsets[name])
                while True:
                    data=reader.read(65536);offsets[name]=reader.tell()
                    if data:
                        fragments[name]+=decoders[name].decode(data)
                        lines=fragments[name].split('\n');fragments[name]=lines.pop()
                        for line in lines:print(line.rstrip('\r'),file=stream,flush=True)
                    if not data or not final:break
            if final:
                fragments[name]+=decoders[name].decode(b'',final=True)
                if fragments[name]:print(fragments[name].rstrip('\r'),file=stream,flush=True);fragments[name]=''
    def log_exception():
        raw=traceback.format_exc()
        try:
            with (out/'stderr.log').open('a',encoding='utf8') as log:log.write(raw)
        except OSError:
            print(raw,file=sys.stderr,flush=True)
    job=None;proc=None;parent_identity=None;known={};peak=0;status='failed';error=None;cleanup_error=None;cleanup=None;interrupted=False
    cpu_processes=[]
    resource_sample_count=0;resource_publish_count=0;next_resource=started;next_publish=started;next_echo=started;rss=None
    job_memory={}
    own_cpu_start=time.process_time();own_cpu_previous=0.;own_wall_previous=started
    try:
        if parent_record:
            _atomic_record(parent_record,{'status':'starting','progress_file':str(progress_file),'process_result_path':str(out/'process_result.json')})
        publish_progress('supervising_child',child_progress_file=str(progress_file),child_status_file=parent_status_file)
        if os.name=='nt':job=_WindowsJob(memory_bytes)
        def limit_posix():
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(memory_bytes,memory_bytes))
        with (out/'stdout.log').open('w',encoding='utf8') as stdout,(out/'stderr.log').open('w',encoding='utf8') as stderr:
            if deadline is not None and time.monotonic()>=deadline:status='timeout'
            else:
                proc=subprocess.Popen(list(map(str,command)),cwd=str(cwd),env=env,stdout=stdout,stderr=stderr,start_new_session=(os.name!='nt'),preexec_fn=limit_posix if os.name!='nt' else None,creationflags=0x00000004 if os.name=='nt' else 0)
                try:parent_identity=(proc.pid,psutil.Process(proc.pid).create_time())
                except (psutil.NoSuchProcess,psutil.AccessDenied):parent_identity=(proc.pid,None)
                if job:
                    job.assign(proc)
                    job.resume(proc)
                if parent_record:
                    _atomic_record(parent_record,{'status':'running','pid':proc.pid,'create_time':parent_identity[1],
                        'progress_file':str(progress_file),'process_result_path':str(out/'process_result.json')})
                while proc.poll() is None:
                    now=time.monotonic()
                    if now>=next_resource:
                        process_samples=[];publish_due=now>=next_publish
                        try:
                            state,parent,_=_probe_identity(parent_identity,[])
                            children=parent.children(recursive=True) if state=='alive' else []
                            _remember_children(known,children)
                            if parent is None:raise psutil.NoSuchProcess(proc.pid)
                            cpu_processes=[parent]+children
                            rss=0
                            for process in cpu_processes:
                                try:
                                    with process.oneshot():
                                        rss+=process.memory_info().rss
                                        if publish_due:
                                            times=process.cpu_times()
                                            process_samples.append({'pid':process.pid,'create_time':process.create_time(),
                                                'cpu_seconds':times.user+times.system})
                                except psutil.NoSuchProcess:continue
                            peak=max(peak,rss)
                        except psutil.Error:rss=None
                        if publish_due:
                            try:own_rss=psutil.Process().memory_info().rss
                            except psutil.Error:own_rss=None
                            cpu_observation=cpu_accounting.sample(now,job=job,processes=cpu_processes)
                            if job:
                                try:job_memory=job.memory_usage()
                                except OSError as exc:job_memory={'job_memory_error':str(exc)}
                            own_cpu=time.process_time()-own_cpu_start
                            own_cores=(own_cpu-own_cpu_previous)/(now-own_wall_previous) if now>own_wall_previous else None
                            tree_cores=cpu_observation.get('cpu_interval_average_cores')
                            _display_record(resource_file,{'updated_monotonic':now,'tree_rss_mib':None if rss is None else rss/1024**2,
                                'batch_rss_mib':None if rss is None or own_rss is None else (rss+own_rss)/1024**2,
                                **cpu_observation,**job_memory,'processes':process_samples,'supervisor_cpu_seconds':own_cpu,
                                'supervisor_cpu_interval_average_cores':own_cores,
                                'batch_cpu_interval_average_cores':None if tree_cores is None or own_cores is None else tree_cores+own_cores})
                            own_cpu_previous=own_cpu;own_wall_previous=now
                            resource_publish_count+=1;next_publish=now+resource_publish_sec
                        resource_sample_count+=1;next_resource=now+resource_sample_sec
                    if now>=next_echo:
                        echo_logs();next_echo=now+.25
                    heartbeat(rss)
                    if rss is not None and rss>memory_bytes:status='memory_limit';break
                    if deadline is not None and time.monotonic()>=deadline:status='timeout';break
                    time.sleep(min(poll_sec,heartbeat_sec or poll_sec))
                else:
                    interrupted=proc.returncode in (-2,130,-1073741510,3221225786)
                    status='interrupted' if interrupted else ('passed' if proc.returncode==0 else 'failed')
    except KeyboardInterrupt as exc:
        error=repr(exc);status='interrupted';interrupted=True;log_exception()
        print(f'[ERROR] {heartbeat_tag}: interrupted; log={out/"stderr.log"}',file=sys.stderr,flush=True)
    except Exception as exc:
        error=repr(exc);status='supervisor_error';log_exception()
        print(f'[ERROR] {heartbeat_tag}: {error}; log={out/"stderr.log"}',file=sys.stderr,flush=True)
    finally:
        execution_status=status
        if proc is not None:
            try:cleanup=_kill_tree(proc,known,job=job,parent_identity=parent_identity)
            except BaseException as exc:
                cleanup_error=repr(exc);error=(error or '')+' cleanup: '+cleanup_error
                if isinstance(exc,KeyboardInterrupt):interrupted=True
                log_exception()
        if job:
            # Query before closing the owned Job; these counters survive child exit.
            cpu_observation=cpu_accounting.sample(time.monotonic(),job=job)
            try:job_memory=job.memory_usage()
            except OSError as exc:job_memory={'job_memory_error':str(exc)}
            try:job.close()
            except Exception as exc:
                cleanup_error=repr(exc);error=(error or '')+' job cleanup: '+cleanup_error;log_exception()
        else:
            cpu_observation=cpu_accounting.sample(time.monotonic(),processes=cpu_processes)
    alive=[] if cleanup is None else cleanup['remaining_pids']
    unconfirmed=[] if cleanup is None else cleanup['unconfirmed_pids']
    if proc is not None and proc.poll() is None:alive.append(proc.pid)
    cleanup_ok=cleanup_error is None and (proc is None or (cleanup is not None and cleanup['parent_reaped'] and cleanup['job_confirmed_empty'] and not cleanup['errors'] and not alive and not unconfirmed))
    if not cleanup_ok:
        status='cleanup_failed'
        if cleanup_error is None:error=(error or '')+' cleanup: process exit or Job drain could not be confirmed'
    child_reports=_child_records(child_status_file)
    # Keep every child result: a later successful shard must not erase failure.
    priorities=('cleanup_failed','interrupted','memory_limit','timeout','supervisor_error','failed')
    child_failures=[r for r in child_reports if r.get('status') in priorities]
    child_status=min(child_failures,key=lambda r:priorities.index(r['status'])) if child_failures else None
    child_failure=status=='failed' and child_status is not None
    if child_failure:status=child_status['status']
    if status=='failed' and not child_failure and (out/'stderr.log').exists() and any(s in (out/'stderr.log').read_text(encoding='utf8',errors='replace') for s in ('MemoryError','std::bad_alloc','Cannot allocate memory')):status='memory_limit'
    interrupted=interrupted or status=='interrupted'
    reason={'passed':'completed','interrupted':'user_interrupted','memory_limit':'memory_limit','timeout':'timeout',
            'cleanup_failed':'cleanup_failed','supervisor_error':'supervisor_error','failed':'execution_failed'}[status]
    if child_failure and status=='timeout':reason='child_timeout'
    result={'command':list(map(str,command)),'cwd':str(cwd),'status':status,'execution_status':execution_status,'interrupted':interrupted,'returncode':None if proc is None else proc.returncode,'elapsed_seconds':time.monotonic()-started,'peak_tree_rss_bytes':peak,'memory_limit_bytes':memory_bytes,'timeout_seconds':timeout_sec,'effective_deadline_monotonic':deadline,'inherited_deadline_monotonic':inherited_deadline,'threads':threads,'workers':1,'memory_enforcement':'windows_job_commit_and_rss' if os.name=='nt' else 'per_process_address_space_hard_and_aggregate_rss_poll_50ms','all_processes_reaped':cleanup_ok,'remaining_pids':sorted(set(alive)),'unconfirmed_pids':sorted(set(unconfirmed)),'cleanup':cleanup,'cleanup_error':cleanup_error,'error':error}
    result.update(cpu_observation);result.update(job_memory)
    result.update(resource_sample_sec=resource_sample_sec,poll_sec=poll_sec,resource_sample_count=resource_sample_count,
                  resource_publish_sec=resource_publish_sec,resource_publish_count=resource_publish_count,
                  resource_file=str(resource_file),supervisor_cpu_seconds=time.process_time()-own_cpu_start)
    if os.name!='nt':result['memory_enforcement']='per_process_address_space_hard_and_aggregate_rss_sampled'
    result.update(blas_threads=blas_threads,thread_environment={k:env.get(k) for k in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    result.update(termination_reason=reason,child_process_report=child_status.get('process_result_path') if isinstance(child_status,dict) else None,
                  child_status=child_status,child_process_reports=child_reports,progress_file=str(progress_file),
                  heartbeat_file=str(out/'supervisor_heartbeat.jsonl'),heartbeat_count=heartbeat_count,
                  process_result_path=str(out/'process_result.json'))
    echo_logs(final=True)
    # Readers keep logical filenames; closed verbose streams are byte-exact gzip.
    # Never compress a live stream while the echo reader still holds offsets.
    from result_storage import finalize_log,storage_record
    result['log_storage']={}
    result['storage_status']='complete' if cleanup_ok else 'deferred_unconfirmed_process_exit'
    for name in ('stdout.log','stderr.log','supervisor_heartbeat.jsonl'):
        if not cleanup_ok:
            # An unconfirmed descendant can retain inherited log handles after
            # ours close. Keep its pathname and bytes available for later writes.
            if (out/name).exists():
                result['log_storage'][name]={'path':str(out/name),'compression':'none',
                    'status':'deferred_unconfirmed_process_exit'}
            continue
        try:
            if (out/name).exists():result['log_storage'][name]=storage_record(finalize_log(out/name))
        except (OSError,ValueError) as exc:
            result['log_storage'][name]={'status':'failed','error':repr(exc)}
            result.update(storage_status='failed',status='failed',termination_reason='storage_failed')
    if parent_record:
        try:
            _atomic_record(parent_record,{'status':result['status'],'execution_status':execution_status,'returncode':result['returncode'],
                'termination_reason':result['termination_reason'],'process_result_path':str(out/'process_result.json'),'progress_file':str(progress_file)})
        except Exception as exc:
            # Logs are finalized; retain the complete traceback in the required result.
            result['registration_error_traceback']=traceback.format_exc()
            print(f'[ERROR] {heartbeat_tag}: result registration failed: {exc}; log={out/"process_result.json"}',file=sys.stderr,flush=True)
            result.update(status='supervisor_error',termination_reason='supervisor_error',
                error=(result['error'] or '')+' result registration: '+repr(exc))
    try:
        _atomic_record(out/'process_result.json',result)
    except Exception as exc:
        print(f'[ERROR] {heartbeat_tag}: required result write failed: {exc}; result={out/"process_result.json"}',file=sys.stderr,flush=True)
        raise
    publish_progress('child_finished',child_status_file=parent_status_file,child_status=result['status'])
    return result

