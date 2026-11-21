"""AE supervision and append-only attempt journal. No automatic retry."""
from __future__ import annotations
import hashlib,json,math,os,pickle,secrets,sys,time,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parent
_ACTIVE=None


class BudgetExceeded(RuntimeError):
    def __init__(self,message,status='budget_limited'):
        super().__init__(message);self.status=status


class RunBudget:
    def __init__(self,timeout=None,max_power=None,journal=None):
        if timeout is not None and (isinstance(timeout,bool) or not math.isfinite(float(timeout)) or float(timeout)<=0):
            raise ValueError('overall timeout must be None or positive and finite')
        self.started=time.monotonic()
        self.deadline=None if timeout is None else self.started+float(timeout)
        outer=os.environ.get('VRQO_DEADLINE_MONOTONIC')
        if outer:
            outer=float(outer)
            if not math.isfinite(outer):raise ValueError('inherited deadline must be finite')
            self.deadline=outer if self.deadline is None else min(self.deadline,outer)
        if max_power is not None and (isinstance(max_power,bool) or int(max_power)!=max_power or max_power<0):raise ValueError('max_power must be nonnegative integer or None')
        self.max_power=None if max_power is None else int(max_power)
        self.journal=Path(journal) if journal else None;self.events=[]
    def record(self,stage,**fields):
        item={'stage':stage,'elapsed_seconds':time.monotonic()-self.started,**fields}
        self.events.append(item)
        if self.journal:
            with self.journal.open('a',encoding='utf-8') as f:
                f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        return item
    def check(self,stage,power=None):
        if self.deadline is not None and time.monotonic()>=self.deadline:
            self.record(stage,status='timeout');raise BudgetExceeded('overall AE deadline exhausted','timeout')
        if power is not None and (power<0 or int(power)!=power):raise ValueError('power must be a nonnegative integer')
        if power is not None and self.max_power is not None and power>self.max_power:
            self.record(stage,status='budget_limited',requested_power=power,max_power=self.max_power)
            raise BudgetExceeded(f'power {power} exceeds cap {self.max_power}')
    def summary(self):
        submitted=[e for e in self.events if e.get('stage')=='sample_submit']
        completed=[e for e in self.events if e.get('stage')=='sample_complete']
        return {'grover_iterations_submitted':sum(e['shots']*e['power'] for e in submitted),
            'A_equivalent_calls_submitted':sum(e['shots']*(2*e['power']+1) for e in submitted),
            'grover_iterations_completed':sum(e['shots']*e['power'] for e in completed),
            'A_equivalent_calls_completed':sum(e['shots']*(2*e['power']+1) for e in completed),
            'shots_submitted':sum(e['shots'] for e in submitted),'shots_completed':sum(e['shots'] for e in completed),
            'max_A_equivalent_calls_per_shot':max((2*e['power']+1 for e in submitted),default=0),
            'failed_attempt_work':'actual partial work unknown; submitted counts are upper budgets, not completed work',
            'retries':0}


def budget(timeout=None,max_power=None):
    return _ACTIVE if _ACTIVE is not None else RunBudget(timeout,max_power)


def run_ae_worker(kind,payload,*,timeout_sec=None,output_dir=None,memory_bytes=None):
    """Always create a real supervised subprocess; env flags are not authority."""
    from runtime_identity import validate_output_path,write_json_new,software_versions,source_manifest
    from result_storage import finalize_log, open_result, read_json, result_exists
    from supervised import run_limited
    timing=RunBudget(timeout=timeout_sec)
    started=timing.started
    timeout_sec=None if timeout_sec is None else float(timeout_sec)
    payload=dict(payload)
    threads=(payload.get('backend_options') or {}).get('max_parallel_threads',1)
    if type(threads) is not int or threads<=0:raise ValueError('max_parallel_threads must be a positive integer')
    try:blas_threads=int(os.environ.get('VRQO_BLAS_THREADS','1'))
    except ValueError as exc:raise ValueError('VRQO_BLAS_THREADS must be a positive integer') from exc
    if blas_threads<=0:raise ValueError('VRQO_BLAS_THREADS must be a positive integer')
    heartbeat=payload.get('sampler_heartbeat_sec')
    heartbeat=30. if heartbeat is None else float(heartbeat)
    if payload.get('seed') is None:payload['seed']=secrets.randbits(32)
    out=validate_output_path(output_dir or ROOT/'outputs'/'ae_runs'/uuid.uuid4().hex)
    out.mkdir(parents=True,exist_ok=False)
    request=out/'request.pkl'
    with request.open('xb') as f:pickle.dump({'kind':kind,'payload':payload,'timeout_sec':timeout_sec},f,protocol=5)
    # Serialization consumes explicit/inherited time, but creates no deadline
    # when neither the caller nor its supervisor requested one.
    remaining=None if timing.deadline is None else timing.deadline-time.monotonic()
    prelaunch_timeout=remaining is not None and remaining<=0
    if prelaunch_timeout:
        process={'status':'timeout','all_processes_reaped':True,'process_started':False,'elapsed_seconds':time.monotonic()-started}
    else:
        process=run_limited([sys.executable,'-B','-m','ae_runtime',str(request)],cwd=ROOT,output_dir=out/'process',timeout_sec=remaining,memory_bytes=memory_bytes,threads=threads,blas_threads=blas_threads,
            heartbeat_sec=heartbeat,heartbeat_tag=f'AE {kind}')
    result_path=out/'worker_result.json';value=read_json(result_path) if result_exists(result_path) else {}
    journal=out/'attempts.jsonl';events=[]
    if result_exists(journal):
        with open_result(journal) as handle:
            for line in handle:
                line=line.rstrip('\r\n')
                try:events.append(json.loads(line))
                except json.JSONDecodeError:events.append({'stage':'journal','status':'incomplete_tail','raw':line})
    meta=value.setdefault('meta',{})
    meta.setdefault('backend_settings',None)
    meta.setdefault('measurement_mapping',None)
    if meta['backend_settings'] is None:
        settings=[e['backend_settings'] for e in events if e.get('stage')=='backend_configured']
        if settings:meta['backend_settings']=settings[-1]
    meta['backend_construction_observed']=any(e.get('stage')=='backend_configured' for e in events)
    if process['status']!='passed':
        meta['status']=process['status'];meta['confidence_interval']=None;value['value']=None
    if not value.get('meta',{}).get('status'):meta['status']='failed'
    meta.update({'run_directory':str(out),'supervision':process,'attempts':events,
        'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),
        'overall_elapsed_seconds':time.monotonic()-started,'automatic_retries':0,'attempt_count':len([e for e in events if e.get('stage')=='sample_submit']),
        'run_id':uuid.uuid4().hex,'root_seed':payload['seed'],'seed_derivation':'sha256(str(root_seed):call_id:pub_id:chunk_id), first 4 bytes little-endian; collision => failure',
        'sample_interval_scope':'per (call_id,pub_id), half-open; separate powers measure different distributions',
        'sources':source_manifest(),'software_versions':software_versions()})
    if 'costs' not in meta:
        fallback=RunBudget();fallback.events=events;meta['costs']=fallback.summary()
    if meta['status']!='completed':meta['precision_target_met']=False
    write_json_new(out/'result.json',value)
    if process.get('all_processes_reaped') and result_exists(journal):
        finalize_log(journal)
    if prelaunch_timeout and os.environ.get('VRQO_PARENT_STATUS_FILE'):
        # No inner supervisor exists to report this known timeout to our parent.
        from supervised import _atomic_record
        record=Path(os.environ['VRQO_PARENT_STATUS_FILE']).with_suffix('')/(uuid.uuid4().hex+'.json')
        _atomic_record(record,{'status':'timeout','report_kind':'ae_prelaunch_failure',
            'termination_reason':'prelaunch_timeout','process_started':False,
            'ae_result_path':str(out/'result.json'),'process_result_path':str(out/'result.json')})
    return value


def _worker(request):
    global _ACTIVE
    from runtime_identity import validate_output_path,write_json_new
    from supervised import publish_progress,_atomic_record
    request=validate_output_path(request)
    # Request is a locally generated pickle under the current project's outputs.
    # Never accept an untrusted downloaded pickle through this internal entry.
    with request.open('rb') as f:data=pickle.load(f)
    out=request.parent;p=data['payload'];kind=data['kind']
    _ACTIVE=RunBudget(data['timeout_sec'],p.get('iqae_max_grover_power',p.get('max_grover_power')),out/'attempts.jsonl')
    _ACTIVE.record('worker_start',status='running',kind=kind)
    publish_progress('ae_worker_start',shots_completed=0,shots_total=None,kind=kind,scope='ae_worker')
    try:
        if kind=='iqae':
            from iqae import _run_iqae_in_process
            val,meta=_run_iqae_in_process(**p);result={'value':val,'meta':meta}
        elif kind=='fixed':
            from iqae import _run_fixed_in_process
            result=_run_fixed_in_process(**p)
        elif kind=='rrk':
            from rrk_model import _run_rrk_in_process
            result=_run_rrk_in_process(**p)
        else:raise ValueError('unknown AE worker kind')
    except Exception as exc:
        status='memory_limit' if isinstance(exc,MemoryError) else getattr(exc,'status','failed')
        _ACTIVE.record('worker_error',status=status,error=repr(exc),error_type=type(exc).__name__)
        result={'value':None,'rows':[],'meta':{'status':status,'error':repr(exc),'error_type':type(exc).__name__,
            'backend_settings':None,'measurement_mapping':None,'precision_target_met':False}}
    result.setdefault('meta',{})['costs']=_ACTIVE.summary()
    result['meta']['attempts']=_ACTIVE.events
    write_json_new(out/'worker_result.json',result)
    parent_status=os.environ.get('VRQO_PARENT_STATUS_FILE')
    status=result['meta'].get('status','failed')
    if parent_status and status not in ('completed','budget_limited'):
        # The worker can observe its deadline before the supervisor's next poll.
        # Preserve this typed evidence without overwriting any sibling result;
        # the supervisor still owns the actual exit code and cleanup outcome.
        record=Path(parent_status).with_suffix('')/(uuid.uuid4().hex+'.json')
        _atomic_record(record,{'status':status,'report_kind':'ae_worker_result',
            'termination_reason':'worker_reported_failure',
            'worker_result_path':str(out/'worker_result.json'),
            'process_result_path':str(out/'worker_result.json')})
    publish_progress('ae_worker_finished',shots_completed=result['meta']['costs']['shots_completed'],
        shots_total=None,kind=kind,scope='ae_worker',status=result['meta'].get('status'))
    return 0 if result['meta'].get('status') in ('completed','budget_limited') else 1

if __name__=='__main__':
    sys.modules['ae_runtime']=sys.modules[__name__]
    raise SystemExit(_worker(sys.argv[1]))
