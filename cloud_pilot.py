"""Validation and performance tasks with no default wall deadline; default is a dry run."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata as md
import json
import math
from pathlib import Path
import platform
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from runtime_identity import prepare_run_directory,validate_output_path,source_manifest,digest

CORE_PACKAGES=('qiskit','qiskit-aer','qiskit-algorithms','numpy','scipy','psutil')

TASKS={
 't12_c32':dict(steps=12,profile='compressed',bond=32,shots=16,seed=2026102612,timeout_sec=None,requires=None),
 't15_c32':dict(steps=15,profile='compressed',bond=32,shots=16,seed=2026102615,timeout_sec=None,requires='t12_c32'),
 't15_b32':dict(steps=15,profile='baseline',bond=32,shots=16,seed=2026102715,timeout_sec=None,requires='t15_c32'),
 't15_h32':dict(steps=15,profile='full_history',bond=32,shots=16,seed=2026102815,timeout_sec=None,requires='t15_c32'),
 't12_c128':dict(steps=12,profile='compressed',bond=128,shots=64,seed=2026102912,timeout_sec=None,requires='t12_c32'),
 't15_c128':dict(steps=15,profile='compressed',bond=128,shots=64,seed=2026102915,timeout_sec=None,requires='t15_c32'),
}

def command_for(root,python,task,out,*,threads=2,memory_mib=4096):
    s=TASKS[task]
    return [str(python),'-B',str(Path(root)/'main.py'),'--phase','test','--steps',str(s['steps']),
        '--kernel-profile',s['profile'],'--shots',str(s['shots']),'--seed',str(s['seed']),
        '--global-shot-offset','0','--aer-shot-chunk',str(s['shots']),'--aer-threads',str(threads),
        '--aer-max-mem-mb',str(memory_mib),'--mps-max-bond',str(s['bond']),'--mps-trunc','1e-12',
        '--aer-safe-basis','--debug-measure-work','--test-measure-dir','--test-measure-x',
        '--run-memory-mib',str(memory_mib),
        '--save-compiled-qpy',str(Path(out)/'result'/'compiled.qpy'),
        '--outdir',str(Path(out)/'result')]

def verify_bundle(manifest):
    if manifest is None:
        from qc_versions import PYTHON_VERSION, PACKAGE_VERSIONS
        sources=source_manifest()
        return {'files':sources,'environment':{'python':PYTHON_VERSION,'versions':dict(PACKAGE_VERSIONS)}},digest(sources)
    obj=json.loads(Path(manifest).read_text(encoding='utf8'));bad=[]
    for name,expected in obj['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append(name)
    if bad:raise ValueError('Cloud bundle differs: '+', '.join(bad))
    return obj,digest(obj['files'])

def environment():
    import psutil
    return {'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
        'versions':{p:md.version(p) for p in CORE_PACKAGES},
        'logical_cpus':psutil.cpu_count(),'available_memory_bytes':psutil.virtual_memory().available,
        'total_memory_bytes':psutil.virtual_memory().total,'cgroup':cgroup_limits()}

def software_identity(record,context):
    """Compare required software versions, independent of installation paths."""
    if not isinstance(record,dict) or not isinstance(record.get('python'),str) or not record['python'].strip():
        raise ValueError(context+' environment must record the actual Python version')
    versions=record.get('versions')
    if not isinstance(versions,dict) or any(not isinstance(versions.get(p),str) or not versions[p].strip() for p in CORE_PACKAGES):
        raise ValueError(context+' environment must record every actual core package version')
    return {'python':record['python'],'versions':{p:versions[p] for p in CORE_PACKAGES}}

def require_same_environment(record,current,context):
    recorded=software_identity(record,context)
    observed=software_identity(current,'Current')
    if recorded!=observed:
        raise ValueError(context+' environment differs from current Python/core packages; run validation again in this environment. '
                         +'Recorded: '+json.dumps(recorded)+'; current: '+json.dumps(observed))

def cgroup_limits():
    if not sys.platform.startswith('linux'):return None
    entry=next((line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')),None)
    if entry is None:return {'supported':False,'reason':'cgroup v2 required'}
    group=Path('/sys/fs/cgroup')/entry.lstrip('/')
    keys=('memory.max','memory.swap.max','pids.max','cpu.max')
    return {'supported':True,'path':str(group),**{k:(group/k).read_text().strip() if (group/k).exists() else 'missing' for k in keys}}

def validate_cgroup(values):
    if not values or not values.get('supported'):raise ValueError('Linux execution requires the provided systemd cgroup wrapper')
    try:
        memory=int(values['memory.max']);pids=int(values['pids.max']);swap=int(values['memory.swap.max'])
        quota,period=map(int,values['cpu.max'].split())
    except (ValueError,KeyError,AttributeError) as exc:raise ValueError('Finite Linux memory/CPU/task cgroup limits are required') from exc
    if not 0<memory<=4*1024**3 or not 0<pids<=64 or swap!=0 or not 0<quota/period<=2:
        raise ValueError('Cloud limits must be memory<=4GiB, swap=0, tasks<=64, CPU<=2 equivalents')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',default=str(ROOT));ap.add_argument('--manifest',default=None,help='optional historical manifest; current identity is derived from root Python sources')
    ap.add_argument('--task',choices=['env']+list(TASKS),required=True);ap.add_argument('--outdir',required=True)
    ap.add_argument('--preflight-report');ap.add_argument('--prior-report')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--threads',type=int,help='positive thread limit; validation default 1, performance default 2')
    ap.add_argument('--memory-mib',type=int,help='positive memory limit in MiB; explicit values are used without a local-mode clamp')
    ap.add_argument('--group',action='append',help='legacy development option; use run_local_validation.py directly')
    ap.add_argument('--timeout',type=float,help='optional environment-check timeout; default no time limit')
    a=ap.parse_args()
    if Path(a.project_root).resolve()!=ROOT:ap.error('Run the cloud_pilot.py contained in the selected project root')
    if a.execute and a.dry_run:ap.error('Choose execute or dry-run')
    if a.threads is not None and a.threads<=0:ap.error('--threads must be positive')
    if a.memory_mib is not None and a.memory_mib<=0:ap.error('--memory-mib must be positive')
    if a.timeout is not None and (not math.isfinite(a.timeout) or a.timeout<=0):ap.error('--timeout must be a finite positive number')
    if a.group:ap.error('--group is a development option; run run_local_validation.py directly when tests are installed')
    if a.task!='env' and (a.group or a.timeout is not None):ap.error('--group and --timeout apply only to validation; performance tasks have no default wall deadline')
    bundle,bundle_hash=verify_bundle(a.manifest);env=environment()
    threads=a.threads if a.threads is not None else (1 if a.task=='env' else 2)
    memory_mib=a.memory_mib if a.memory_mib is not None else (max(1,min(4096,env['available_memory_bytes']//(4*1024**2))) if a.task=='env' else 4096)
    from config import default_config,validate_quantum_config
    from unitary_circuit import resolve_kernel_profile
    cfg=default_config();out=validate_output_path(a.outdir)
    if a.task=='env':
        command=[sys.executable,'-B',str(ROOT/'run.py'),'check-env','--check','--outdir',str(out/'environment')]
    else:command=command_for(ROOT,sys.executable,a.task,out,threads=threads,memory_mib=memory_mib)
    if a.task!='env':
        validate_quantum_config(cfg,TASKS[a.task]['steps'])
        from main import build_argparser
        build_argparser().parse_args(command[3:])
    plan={'task':a.task,'dry_run':not a.execute,'bundle_digest':bundle_hash,'environment':env,'command':command,
        'config':asdict(cfg),'settings':None if a.task=='env' else TASKS[a.task],
        'actual_profile':None if a.task=='env' else asdict(resolve_kernel_profile(kernel_profile=TASKS[a.task]['profile'])),
        'observables':[] if a.task=='env' else ['integer_four_terminal_counts','measured_work_subset','direction','x'],
        'sample_interval':None if a.task=='env' else [0,TASKS[a.task]['shots']],
        'threads':threads,'workers':1,'memory_mib':memory_mib,
        'timeout_seconds':a.timeout if a.task=='env' else TASKS[a.task]['timeout_sec'],
        'group_timeout_seconds':None,'full_suite':False if a.task=='env' else None,
        'full_science_validation':False,'legacy':True,
        'purpose':'environment check only; no scientific validation' if a.task=='env' else 'performance pilot only; not precision certification'}
    if not a.execute:
        print(json.dumps(plan,indent=2));return 0
    expected=bundle['environment']
    if sys.platform.startswith('linux'):validate_cgroup(env['cgroup'])
    if a.task!='env':
        require_same_environment(expected,env,'Required bundle')
        if env['logical_cpus']<threads or env['available_memory_bytes']<max(16*1024**3,memory_mib*1024**2):raise ValueError('Pilot requires enough CPUs for requested threads and currently available memory >= max(16 GiB, requested memory); do not oversubscribe')
        if not a.preflight_report:raise ValueError('Provide the successful cloud env cloud_preflight.json')
        ready=json.loads(Path(a.preflight_report).read_text(encoding='utf8'))
        if ready.get('all_passed') is not True or ready.get('full_suite') is not True or ready.get('full_science_validation') is False or ready.get('bundle_digest')!=bundle_hash:raise ValueError('A full scientific validation preflight must have passed for this bundle; environment checks alone are insufficient')
        require_same_environment(ready.get('environment'),env,'Full validation preflight')
        need=TASKS[a.task]['requires']
        if need:
            if not a.prior_report:raise ValueError('Provide successful prior pilot_report.json for '+need)
            prior=json.loads(Path(a.prior_report).read_text(encoding='utf8'))
            if prior.get('task')!=need or prior.get('status')!='passed' or prior.get('bundle_digest')!=bundle_hash:raise ValueError('Previous bounded pilot did not pass for this bundle: '+need)
            require_same_environment(prior.get('environment'),env,'Previous pilot')
    out=prepare_run_directory(out);(out/'launch_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    from supervised import run_limited
    if a.task=='env':
        print(f"START environment check: timeout_seconds={a.timeout!r}, threads={threads}, memory_mib={memory_mib}",flush=True)
        result=run_limited(command,cwd=ROOT,output_dir=out/'supervision',timeout_sec=a.timeout,
                           memory_bytes=memory_mib*1024**2,threads=threads,echo_output=True)
        local=out/'environment/environment.json'
        local_report=json.loads(local.read_text(encoding='utf8')) if local.exists() else {}
        interrupted=bool(result.get('interrupted') or result.get('returncode')==130 or local_report.get('interrupted'))
        passed=(result['status']=='passed' and local_report.get('matches') is True
                and local_report.get('imports_ok') is True
                and (local_report.get('dependency_check') or {}).get('returncode')==0 and not interrupted)
        status='passed' if passed else ('interrupted' if interrupted and result['status']!='cleanup_failed' else result['status'])
        if status=='passed' and not passed:status='failed'
        report={'task':'env','status':status,'all_passed':passed,'full_suite':False,
                'full_science_validation':False,'validation_scope':'environment_only','legacy':True,'interrupted':interrupted,
                'bundle_digest':bundle_hash,'process':result,'environment':env,
                'environment_report':str(local.resolve()),'timeout_seconds':a.timeout,'group_timeout_seconds':None,
                'threads':threads,'memory_mib':memory_mib}
        report_path=out/'cloud_preflight.json'
        report_path.write_text(json.dumps(report,indent=2),encoding='utf8')
        print(f"END environment check: status={status}, all_passed={passed}, full_science_validation=False, report={report_path.resolve()}",flush=True)
        print(f"Logs: {(out/'supervision').resolve()}",flush=True)
        return 130 if interrupted else (0 if passed else 1)
    result=run_limited(command,cwd=ROOT,output_dir=out/'supervision',timeout_sec=None,memory_bytes=memory_mib*1024**2,threads=threads,echo_output=True)
    report={'task':a.task,'bundle_digest':bundle_hash,**result,'environment':env,'precision_certified':False}
    raw_path=out/'result/results.json'
    from result_storage import read_json, result_exists
    report.update(counts_complete=False,raw_result_path=str(raw_path),raw_result_error=None)
    if result_exists(raw_path):
        try:
            from runtime_identity import validate_completed_result
            raw=read_json(raw_path)
            n=validate_completed_result(raw)
            if n!=TASKS[a.task]['shots'] or raw.get('steps')!=TASKS[a.task]['steps']:
                raise ValueError('Completed counts do not match the requested task')
            report['counts_complete']=True
        except (OSError,ValueError,TypeError) as exc:report['raw_result_error']=repr(exc)
    else:report['raw_result_error']='No complete results.json was produced'
    if report['status']=='passed' and not report['counts_complete']:
        report.update(status='failed',termination_reason='incomplete_results')
    (out/'pilot_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True);return 0 if report['status']=='passed' else (130 if report['status']=='interrupted' else 1)

if __name__=='__main__':raise SystemExit(main())
