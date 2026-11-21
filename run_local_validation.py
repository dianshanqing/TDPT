"""Development checks in a disposable system-temp checkout; keep one summary."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata as md
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

from supervised import run_limited

ROOT=Path(__file__).resolve().parent
DEFAULT_GROUPS=(
    'tests.test_reference','tests.test_reference_limits','tests.test_geometry_config',
    'tests.test_transport_circuit','tests.test_ae_quantum',
    'pytest:tests/test_revision_reference.py','pytest:tests/test_compact_reference.py',
    'tests.test_root_runtime','tests.test_revision_entrypoint','tests.test_compact_identity',
    'tests.test_revision_quantum','tests.test_validation_control','tests.test_run_request',
    'tests.test_windows_cmd','tests.test_entrypoints','tests.test_mps_probe_controls',
    'tests.test_mps_log_storage','tests.test_cpu_accounting','tests.test_svd_calibration',
    'tests.test_run_control','tests.test_supervision_layout','tests.test_statistical_summary',
    'tests.test_ae_public','tests.test_engineering',
    'tests.test_root_reference_modules','tests.test_cloud_handoff',
    'tests.test_rare_precision','tests.test_ae','tests.test_ae_unlimited',
    'tests.test_cleanup','tests.test_supervision','tests.test_supervisor_progress',
    'tests.test_cli_integration','tests.test_output_entries',
    'tests.test_numerical_runtime','tests.test_mps_performance','tests.test_performance_request',
    'tests.test_numerical_audit','tests.test_batch_runtime',
    'tests.test_precision_acceptance','tests.test_relative_iqae','tests.test_precision_entry',
    'tests.test_sigma_consumers','tests.test_iqae_resources',
)

# Classification describes evidence, not an inflated sum of equally meaningful tests.
GROUP_EVIDENCE={
    'tests.test_reference':['independent_complex_gates','independent_classical_recursion'],
    'tests.test_transport_circuit':['actual_transport_gates','joint_distribution','step_boundary_work','terminal_freeze'],
    'tests.test_ae_quantum':['actual_transport_inverse','actual_target_marking','actual_grover_relation'],
    'tests.test_revision_quantum':['actual_three_model_profiles','geometry_truth_table','mock_failure_controls'],
    'pytest:tests/test_revision_reference.py':['independent_hand_derived_classical','input_contracts'],
    'pytest:tests/test_compact_reference.py':['independent_decimal','physical_refinement_mapping'],
    'tests.test_mps_probe_controls':['actual_small_transport_mps','instrumented_counts_switch','mock_backend_failure'],
    'tests.test_svd_calibration':['actual_one_and_two_qubit_calibration','separate_accuracy_status'],
    'tests.test_rare_precision':['actual_small_rotations','accuracy_decision_contracts'],
    'tests.test_ae_public':['actual_small_public_sampler','sample_budget_bookkeeping_not_coverage'],
    'tests.test_cli_integration':['actual_small_mps_cli','compiled_cache_identity','sample_completeness'],
    'tests.test_engineering':['identity_contracts','mock_shard_merge','parameter_and_failure_controls'],
    'tests.test_ae':['mock_ae_execution','likelihood_truth_isolation'],
    'tests.test_ae_unlimited':['mock_budget_contracts','process_reaping'],
    'tests.test_mps_performance':['actual_small_prefix_controls','mock_resource_and_comparison_controls'],
    'tests.test_numerical_audit':['actual_small_transport_readouts','independent_complex_gates','bounded_pruning_with_error_ledger'],
    'tests.test_batch_runtime':['preset_budget_contracts','mock_resource_controls_real_acceptance_is_root_copy'],
}


def _tested_manifest(root):
    """Hash the actual copied inputs, including launchers and development tests."""
    paths=list(root.glob('*.py'))+list(root.glob('*.cmd'))
    for folder,pattern in [('tests','*.py'),('tools','*.py'),('configs','*.json')]:
        paths.extend((root/folder).rglob(pattern))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def _scientific_observations(copy):
    """Retain compact scalar evidence; discard disposable circuit/log copies."""
    from result_storage import logical_path, open_result, read_json, result_files
    # run_limited assigns each worker's VRQO_OUTPUT_DIR to its own group directory.
    base=copy/'outputs/_validation_groups';report={}
    for filename in ('transport_science.jsonl','model_science.jsonl'):
        paths=result_files(base, '*/'+filename)
        if paths:
            report[filename]=[]
            for path in paths:
                with open_result(path) as handle:
                    report[filename].extend(json.loads(line) for line in handle)
    for path in result_files(base, '*/quantum_ae_*.json'):
        data=read_json(path)
        name=logical_path(path).name
        report[name]={key:data[key] for key in ('profile','steps','A_fingerprint',
            'inverse_amplitude_error','inverse_tail_probability','probability_tolerance')}
        report[name]['checks']=[{'target':row['target'],'iterations':row['iterations'],
            'all_registers_reflected':row['all_registers_reflected']} for row in data['checks']]
    return report


@contextmanager
def _temporary_runtime():
    # Copy only source and development fixtures, never scientific outputs.
    with tempfile.TemporaryDirectory(prefix='paper1 dev tests ') as td:
        base=Path(td).resolve();copy=base/'runtime';copy.mkdir();(copy/'outputs').mkdir()
        for pattern in ('*.py','*.cmd'):
            for source in ROOT.glob(pattern):shutil.copy2(source,copy/source.name)
        for folder,pattern in [('tests','*.py'),('tools','*.py'),('configs','*.json')]:
            for source in (ROOT/folder).rglob(pattern):
                target=copy/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
        yield base,copy


@contextmanager
def _temporary_output_contract(copy):
    # Scope the developer supervisor to this owned temporary checkout. Restore
    # the normal production output protection even on interruption or failure.
    import runtime_identity
    previous=runtime_identity.PROJECT_ROOT
    runtime_identity.PROJECT_ROOT=copy
    try:yield
    finally:runtime_identity.PROJECT_ROOT=previous


def _run_groups(copy, groups, *, timeout, threads, memory_bytes):
    rows=[];interrupted=False;control_error=None
    env=dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',
             PYTHONWARNINGS='ignore::DeprecationWarning')
    fixture_root=copy/'outputs/test_runs';fixture_root.mkdir(parents=True,exist_ok=True)
    env['VRQO_OUTPUT_DIR']=str(fixture_root)
    for key in ('VRQO_CMD_REQUEST_JSON','VRQO_RUN_CMD'):env.pop(key,None)
    try:
        for index,group in enumerate(groups):
            started=time.monotonic()
            # Long unittest method selectors otherwise exceed legacy Windows path limits
            # once worker/shard output names are appended. The hash prevents collisions.
            label=re.sub(r'[^a-zA-Z0-9_.-]', '_', group)[:32]
            output=copy/'outputs'/'_validation_groups'/f"{index:02d}_{label}_{hashlib.sha256(group.encode()).hexdigest()[:8]}"
            try:
                print(f'START {group} [{index+1}/{len(groups)}]',flush=True)
                if group.startswith('pytest:'):
                    engine='pytest';arguments=[group.split(':',1)[1],'-q','-p','no:cacheprovider',
                        '--basetemp',str(output/'pytest_tmp')]
                else:
                    engine='unittest';arguments=['discover','-s','tests','-v'] if group=='discover' else [group,'-v']
                result=run_limited([sys.executable,'-B','-m',engine,*arguments],
                    cwd=copy,output_dir=output,timeout_sec=timeout,memory_bytes=memory_bytes,
                    threads=threads,env_extra=env,echo_output=True)
            except KeyboardInterrupt:
                error=traceback.format_exc();print(error,file=sys.stderr,flush=True)
                result={'status':'interrupted','interrupted':True,'elapsed_seconds':time.monotonic()-started,'error':error}
            except Exception:
                error=traceback.format_exc();print(error,file=sys.stderr,flush=True)
                result={'status':'runner_error','elapsed_seconds':time.monotonic()-started,'error':error}
            from result_storage import open_result, result_exists
            logs={}
            for name in ('stdout','stderr'):
                path=output/(name+'.log')
                if result_exists(path):
                    with open_result(path, errors='replace') as handle:logs[name]=handle.read()
                else:logs[name]=''
            counts=re.findall(r'Ran (\d+) tests?',logs['stdout']+'\n'+logs['stderr'])
            if group.startswith('pytest:'):
                counts=re.findall(r'(\d+) passed',logs['stdout']+'\n'+logs['stderr'])
            row={'group':group,**result,'tests':int(counts[-1]) if counts else None}
            row['skipped_tests']=re.findall(r"^([^\r\n]+) \.\.\. skipped ['\"]([^\r\n]+)['\"]",logs['stderr'],re.MULTILINE)
            row['evidence_kind']=GROUP_EVIDENCE.get(group,['parameter_process_or_mock_contracts'])
            if row['tests']==0:
                row.update(status='failed',error='Zero tests collected; this is not validation evidence')
            if result['status']!='passed':row.update(logs)
            rows.append(row)
            interrupted=bool(result.get('interrupted')) or result['status']=='interrupted'
            print(f"END {group}: {result['status']} ({result.get('elapsed_seconds',0):.2f}s)",flush=True)
            if interrupted:break
    except KeyboardInterrupt:
        interrupted=True;control_error=traceback.format_exc();print(control_error,file=sys.stderr,flush=True)
    except Exception:
        control_error=traceback.format_exc();print(control_error,file=sys.stderr,flush=True)
    passed=bool(rows) and len(rows)==len(groups) and not interrupted and control_error is None and all(row['status']=='passed' for row in rows)
    failures=[str(row.get(key,'')) for row in rows if row['status']!='passed' for key in ('error','stdout','stderr')]
    if control_error:failures.append(control_error)
    return {'results':rows,'not_run_groups':groups[len(rows):],'interrupted':interrupted,
        'control_error':control_error,'all_passed':passed,'returncode':130 if interrupted else (0 if passed else 1),
        'tests':sum(row['tests'] for row in rows if row['tests'] is not None) if any(row['tests'] is not None for row in rows) else None,
        'failure_details':'\n'.join(failures) if not passed else None}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group',action='append')
    ap.add_argument('--outdir',help='optional compatibility destination for one summary only')
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--memory-mib',type=int,default=None)
    ap.add_argument('--timeout',type=float,default=None,help='optional explicit seconds per test group; default no deadline')
    args=ap.parse_args(argv)
    if not (ROOT/'tests').is_dir():
        ap.error('Development tests are not deployed. Use run.cmd validate --suite three_models_small for production small checks.')
    if args.threads<1 or (args.memory_mib is not None and args.memory_mib<1):ap.error('Positive resource values required')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout<=0):ap.error('--timeout must be finite and positive')
    if os.environ.get('VRQO_DEADLINE_MONOTONIC'):ap.error('Inherited wall deadline is forbidden')
    destination=ROOT/'tests/last_verification.json'
    if args.outdir:
        from runtime_identity import prepare_run_directory
        destination=prepare_run_directory(args.outdir)/'local_validation.json'
    groups=args.group or list(DEFAULT_GROUPS);started=time.monotonic();base=None;tested_sources={};tested_inputs={};observations={}
    memory=None if args.memory_mib is None else args.memory_mib*1024**2
    report={'all_passed':False,'returncode':1,'results':[],'not_run_groups':groups,'tests':None,'interrupted':False}
    print('Running finite developer groups in system temp; no scientific outputs are copied.',flush=True)
    try:
        with _temporary_runtime() as (base,copy),_temporary_output_contract(copy):
            from runtime_identity import source_manifest
            tested_sources=source_manifest(copy)
            tested_inputs=_tested_manifest(copy)
            report=_run_groups(copy,groups,timeout=args.timeout,threads=args.threads,memory_bytes=memory)
            observations=_scientific_observations(copy)
    except (Exception,KeyboardInterrupt) as exc:
        error=traceback.format_exc();print(error,file=sys.stderr,flush=True)
        interrupted=isinstance(exc,KeyboardInterrupt) or report.get('interrupted',False)
        report.update(all_passed=False,returncode=130 if interrupted else 1,interrupted=interrupted,
                      control_error=error,failure_details=(report.get('failure_details') or '')+'\n'+error)
    removed=base is None or not base.exists()
    if not removed:report.update(all_passed=False,returncode=130 if report.get('interrupted') else 1)
    current_inputs=_tested_manifest(ROOT)
    source_unchanged=tested_inputs==current_inputs
    if not source_unchanged:
        report.update(all_passed=False,returncode=130 if report.get('interrupted') else 1)
    report.update(groups=groups,elapsed_seconds=time.monotonic()-started,python=sys.executable,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),temporary_directory=str(base) if base else None,
        temporary_directory_removed=removed,requested_threads=args.threads,threads=args.threads,
        requested_memory_mib=args.memory_mib,memory_limit_bytes=memory,timeout_seconds=args.timeout,
        wall_deadline=None,sources=tested_sources,source_scope='root Python files copied into the tested runtime',
        tested_inputs=tested_inputs,tested_input_scope='root PY/CMD plus copied tests/tools/config fixtures',
        tested_inputs_match_current_source=source_unchanged,
        scientific_observations=observations,
        versions={name:md.version(name) for name in ('qiskit','qiskit-aer','qiskit-algorithms','numpy','scipy','psutil')})
    # Preserve separately completed deployment acceptance when replacing the one
    # development summary; explicit destinations remain immutable new runs.
    if not args.outdir and destination.exists():
        previous=json.loads(destination.read_text(encoding='utf8'))
        if 'root_copy_acceptance' in previous:report['root_copy_acceptance']=previous['root_copy_acceptance']
        if 'prior_failed_verifications' in previous:
            report['prior_failed_verifications']=previous['prior_failed_verifications']
        if previous.get('all_passed') is False:
            report.setdefault('prior_failed_verifications',[]).append({
                key:previous.get(key) for key in ('timestamp_utc','results','failure_details',
                    'sources','tested_inputs','tested_inputs_match_current_source')})
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"VALIDATION END all_passed={report['all_passed']} interrupted={report['interrupted']}",flush=True)
    print(f'Report: {destination}',flush=True)
    return report['returncode']


if __name__=='__main__':raise SystemExit(main())
