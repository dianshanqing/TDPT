"""Shared QC entry point. Resources are arguments, not machine-specific modes.

Windows Anaconda Prompt: conda activate 'QC'
run.cmd validate --suite three_models_small --outdir outputs/validation_001
run.cmd reference --config q_uniform_small --steps 2 --outdir outputs/reference_001
run.cmd mc/test/iqae/mlae [existing main.py arguments]
run.cmd shards [existing shard_runner.py arguments]
run.cmd pilot --task t12_c32 [pilot arguments]
run.cmd summarize --input RAW_RESULT_OR_SHARDS
run.cmd mps_batch --suite mps_batch_small --batch-workers 2 [resource arguments]
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
from qc_versions import PYTHON_VERSION, PACKAGE_VERSIONS
from result_storage import (content_digest, logical_path, read_json, result_exists,
                            result_files, storage_record)


def stable_version_order(actual, target):
    """Compare stable numeric versions; custom/prerelease versions need review."""
    if actual is None:
        return -1
    if not re.fullmatch(r'\d+(?:\.\d+)*', actual):
        return None
    a, b = tuple(map(int, actual.split('.'))), tuple(map(int, target.split('.')))
    width = max(len(a), len(b))
    a, b = a + (0,) * (width - len(a)), b + (0,) * (width - len(b))
    return (a > b) - (a < b)


def environment_record(import_check=False):
    expected = dict(PACKAGE_VERSIONS)
    observed = {}
    for name in expected:
        try:
            observed[name] = md.version(name)
        except md.PackageNotFoundError:
            observed[name] = None
    record = {'environment_name': Path(sys.prefix).name, 'executable': sys.executable,
              'python': platform.python_version(), 'versions': observed,
              'expected_python': PYTHON_VERSION, 'expected_versions': expected,
              'matches': platform.python_version() == PYTHON_VERSION and observed == expected,
              'imports_ok': None, 'dependency_check': None,
              'full_project_validation': 'not_assessed'}
    actual_components = {'python': record['python'], **observed}
    target_components = {'python': PYTHON_VERSION, **expected}
    newer = []
    for name, target in target_components.items():
        actual = actual_components[name]
        order = stable_version_order(actual, target)
        if order is None or order > 0 or (order == 0 and actual != target):
            newer.append({'component': name, 'actual': actual, 'target': target})
    record['newer_or_unreviewed_components'] = newer
    record['update_action'] = ('blocked_unreviewed_version' if newer else
                               'none' if record['matches'] else
                               'update_python_and_packages' if record['python'] != PYTHON_VERSION else
                               'update_packages')
    if import_check:
        errors = {}
        for name in expected:
            try:
                importlib.import_module(name.replace('-', '_'))
            except Exception as exc:
                errors[name] = repr(exc)
        record['imports_ok'] = not errors
        record['import_errors'] = errors
        check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True,
                               text=True, encoding='utf8', errors='replace')
        record['dependency_check'] = {'returncode': check.returncode, 'stdout': check.stdout, 'stderr': check.stderr}
    return record


def require_qc():
    record = environment_record()
    if record['environment_name'] != "'QC'":
        raise RuntimeError("Activate the existing environment with: conda activate 'QC' (quotes are part of its name)")
    if not record['matches']:
        raise RuntimeError('QC versions differ; run setup_qc.cmd. Observed: ' + json.dumps(record))
    return record


def output_directory(requested, purpose):
    from runtime_identity import prepare_run_directory
    return prepare_run_directory(requested or ROOT / 'outputs' / f'{purpose}_{uuid.uuid4().hex[:12]}')


def invoke(path, args):
    saved = sys.argv
    sys.argv = [str(ROOT / path), *args]
    try:
        runpy.run_path(str(ROOT / path), run_name='__main__')
    finally:
        sys.argv = saved
    return 0


def reject_action_override(args, option):
    # argparse also accepts --option=value and unambiguous option abbreviations.
    for value in args:
        key = value.partition('=')[0]
        if key.startswith('--') and len(key) > 2 and option.startswith(key):
            raise ValueError('The fixed action already selects ' + option + '; select it only once')


def check_environment(args):
    ap = argparse.ArgumentParser(description="Check the original 'QC' and record versions before any update.")
    ap.add_argument('--outdir')
    ap.add_argument('--check', action='store_true', help='check only; setup_qc.cmd will not install')
    a = ap.parse_args(args)
    record = environment_record(import_check=True)
    out = output_directory(a.outdir, 'qc_environment') if a.outdir else None
    if out is not None:
        (out / 'environment.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    print(json.dumps(record, indent=2), flush=True)
    if out is not None: print('Environment record:', out / 'environment.json', flush=True)
    if record['environment_name'] != "'QC'":
        return 1
    healthy = record['imports_ok'] and (record.get('dependency_check') or {}).get('returncode') == 0
    if record['matches'] and healthy:
        return 0
    if a.check:
        return 1
    if record.get('newer_or_unreviewed_components'):
        print('A newer or custom version is installed. Review compatibility and update the candidate before installing; no downgrade will run.', file=sys.stderr)
        return 12
    if (record.get('dependency_check') or {}).get('returncode') != 0 or record['matches']:
        print('Dependency or import check failed. Inspect environment.json before changing packages.', file=sys.stderr)
        return 1
    return 10 if record['update_action'] == 'update_python_and_packages' else 11


def reference(args):
    ap = argparse.ArgumentParser(description='Independent classical DP; does not run a quantum simulation.')
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--outdir')
    a = ap.parse_args(args)
    if a.steps <= 0:
        ap.error('--steps must be positive')
    from config import default_config, validate_config
    from reference_engine import reference as independent_reference
    from runtime_identity import source_manifest
    cfg = default_config(); validate_config(cfg)
    out = output_directory(a.outdir, 'reference')
    start = time.monotonic()
    result = independent_reference(cfg, a.steps)
    result.update(phase='independent_reference', status='complete', config=asdict(cfg), environment=environment_record(),
                  sources=source_manifest(), elapsed_seconds=time.monotonic() - start)
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    with (out / 'statistics.csv').open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['steps', *result['order'], 'norm_error'])
        writer.writeheader()
        for row in result['rows']:
            writer.writerow({'steps': row['steps'], **dict(zip(result['order'], row['probabilities'])), 'norm_error': row['norm_error']})
    print('Reference complete:', out, flush=True)
    return 0


def reference_probabilities(path, raw):
    """Read a completed independent DP record after model/row consistency checks."""
    from models import model_from_dict
    from runtime_identity import normalize, check_identity
    record=read_json(path)
    if record.get('phase') not in ('independent_reference','reference') or record.get('status','complete')!='complete':
        raise ValueError('A completed independent reference result is required')
    cfg=record.get('config')
    if record['phase']=='reference':
        check_identity(record.get('identity'))
        if record['identity']['phase']!='reference': raise ValueError('Reference identity phase mismatch')
        cfg=record['identity']['config']
        if normalize(record.get('config'))!=normalize(cfg):
            raise ValueError('Reference config disagrees with its identity')
    try:
        model=model_from_dict(cfg)
    except (TypeError,KeyError,AttributeError,ValueError) as exc:
        raise ValueError('Reference must contain a complete valid transport config') from exc
    if normalize(cfg)!=normalize(asdict(model)) or normalize(cfg)!=raw['identity']['config']:
        raise ValueError('Reference config differs from the raw run identity')
    extra=raw['identity'].get('extra') or {}
    steps=raw.get('steps')
    if type(steps) is not int or steps<=0 or extra.get('steps')!=steps:
        raise ValueError('Raw steps disagree with run identity')
    if extra.get('disable_detector') or extra.get('disable_boundary'):
        raise ValueError('Independent reference does not describe disabled detector/boundary overrides')
    order=record.get('order')
    if not isinstance(order,list) or not all(isinstance(x,str) for x in order):
        raise ValueError('Reference terminal order is missing')
    channels=['survive' if x=='alive' else x for x in order]
    if len(channels)!=4 or len(set(channels))!=4 or set(channels)!=set(raw['counts']):
        raise ValueError('Reference must identify each of the four terminal channels once')
    rows=record.get('rows');selected=None
    if not isinstance(rows,list) or not rows:raise ValueError('Reference rows are missing')
    for expected_step,row in enumerate(rows,1):
        if not isinstance(row,dict) or type(row.get('steps')) is not int or row['steps']!=expected_step:
            raise ValueError('Reference steps must form an ordered complete sequence from 1')
        probs=row.get('probabilities');norm_error=row.get('norm_error')
        if (not isinstance(probs,list) or len(probs)!=4 or
            any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in probs)):
            raise ValueError('Reference probabilities must be finite and within [0,1]')
        if (type(norm_error) not in (int,float) or not math.isfinite(norm_error) or not 0<=norm_error<=2e-12 or
            abs(math.fsum(probs)-1)>2e-12):
            raise ValueError('Reference normalization check failed')
        if row['steps']==steps:selected=dict(zip(channels,probs))
    if selected is None:raise ValueError('Reference lacks the raw run step count')
    return selected,{'path':str(logical_path(path).resolve()),'sha256':content_digest(path)['sha256'],
                     'storage':storage_record(path),
                     'steps':steps,'phase':record['phase'],'config_verified':True,
                     'normalization_tolerance':2e-12,'order_in_record':order}


def summarize(args):
    ap = argparse.ArgumentParser(description='Validate raw runs/shards and write one result JSON plus a statistics CSV.')
    ap.add_argument('--input', '--in', required=True, help='raw results.json[.gz], its directory, or directory containing shard_*/results.json[.gz]')
    ap.add_argument('--outdir', '--out')
    ap.add_argument('--reference',help='independent result.json produced by run.cmd reference; config and steps must match')
    ap.add_argument('--family-size',type=int,default=4,help='prespecified family size for simultaneous 95%% intervals; use 8 for joint T12/T15 four-channel reporting')
    a = ap.parse_args(args)
    source = Path(a.input).resolve()
    if not a.outdir:
        a.outdir=str((source if source.is_dir() else source.parent)/'summary')
    probe = source/'mps_probability_probe.json' if source.is_dir() else source
    if result_exists(probe) and logical_path(probe).name == 'mps_probability_probe.json':
        return summarize_probe(probe, a.outdir)
    if result_exists(source):
        files = [source]
    elif result_exists(source / 'results.json'):
        files = [source / 'results.json']
    else:
        files = result_files(source, 'shard_*/results.json')
    if not files:
        ap.error('No original results.json found. Supply the raw run or shard directory.')
    first = read_json(files[0])
    phase = first.get('phase')
    from merge_shards import merge_test, merge_mc
    if phase not in ('test', 'mc'):
        ap.error('Only raw test/MC counts can be combined; AE records retain their own likelihood analysis.')
    # Existing merge functions validate every identity/count/interval and reject missing or duplicate samples.
    merged = (merge_test if phase == 'test' else merge_mc)([str(p) for p in files])
    from stats_utils import binom_se, sampling_intervals
    counts = merged['counts']; samples = sum(counts.values())
    statistics=sampling_intervals(counts,family_size=a.family_size)
    reference=None
    if a.reference:
        from runtime_identity import normalize
        if 'cfg' in first and normalize(first['cfg'])!=merged['identity']['config']:
            raise ValueError('Raw config disagrees with its identity')
        reference,provenance=reference_probabilities(a.reference,merged)
        statistics['reference_record']=provenance
    for terminal,row in statistics['channels'].items():
        p=None if reference is None else reference[terminal]
        row.update(reference=p,signed_difference=None if p is None else row['probability']-p,
                   difference_interval=None if p is None else [x-p for x in row['confidence_interval']])
    out = output_directory(a.outdir, 'summary')
    merged['summary_scope'] = statistics['scope']
    merged['sampling_statistics']=statistics
    merged['precision_certified']=False
    (out / 'result.json').write_text(json.dumps(merged, indent=2, allow_nan=False), encoding='utf8')
    with (out / 'statistics.csv').open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['terminal', 'count', 'samples', 'probability', 'standard_error',
            'ci_lower','ci_upper','reference','signed_difference','difference_ci_lower','difference_ci_upper',
            'confidence','family_size','precision_certified'])
        writer.writeheader()
        for terminal, count in counts.items():
            probability = count / samples
            row=statistics['channels'][terminal];difference=row['difference_interval']
            writer.writerow(dict(terminal=terminal, count=count, samples=samples,
                probability=probability, standard_error=binom_se(probability, samples),
                ci_lower=row['confidence_interval'][0],ci_upper=row['confidence_interval'][1],
                reference=row['reference'],signed_difference=row['signed_difference'],
                difference_ci_lower=None if difference is None else difference[0],
                difference_ci_upper=None if difference is None else difference[1],
                confidence=statistics['confidence'],family_size=statistics['family_size'],precision_certified=False))
    print('Summary complete:', out, flush=True)
    return 0


def summarize_probe(source, outdir):
    """Recompute scientific criteria from stored raw marginals, never counts."""
    from rare_precision import compare_rare_probability
    from validation import _probe_accuracy_status
    data = read_json(source)
    rows=[]
    for r in data['rows']:
        row={k:r.get(k) for k in ('case','status','bond','truncation_threshold','original_A_fingerprint','compiled_A_fingerprint')}
        row['historical_accuracy_labels']={k:r.get(k) for k in ('rare_accuracy_status','pilot_accuracy_status','strict_status')}
        if r.get('status') == 'complete':
            p,ref=r.get('probabilities'),r.get('reference')
            def detector(values):
                return values[1] if isinstance(values,(list,tuple)) and len(values)==4 else None
            p_d,ref_d=detector(p),detector(ref)
            try:comparison=compare_rare_probability(p_d,ref_d)
            except (TypeError,ValueError):comparison=None
            missing=p is None or not isinstance(p,(list,tuple)) or len(p)!=4 or p_d is None
            row.update(_probe_accuracy_status(p,ref,r.get('work_any_nonzero_union_upper_bound'),
                work_bit_probabilities=r.get('work_bit_probabilities')))
            row.update(p_D=p_d if isinstance(p_d,(int,float)) and math.isfinite(p_d) else None,
                reference_D=ref_d if isinstance(ref_d,(int,float)) and math.isfinite(ref_d) else None,
                relative_error=comparison.get('relative_error') if comparison else None,
                execution_status='complete',rare_comparison=comparison,
                raw_probability_representation=repr(p),raw_reference_representation=repr(ref),
                raw_evidence_location='rows/'+str(len(rows)),work_status=row['work_accuracy_status'])
            if missing:row.update(rare_accuracy_status='not_assessed',pilot_accuracy_status='not_assessed')
        else:
            row.update(execution_status='failed',rare_accuracy_status='not_assessed',pilot_accuracy_status='not_assessed',strict_status='not_assessed',
                numerical_validity_status='not_assessed',scientific_certification_passed=False,error=r.get('error'))
        rows.append(row)
    counts=[{k:r.get(k) for k in ('case','status','executed','shots','sampling_seconds','decoded_status_counts','reason')} for r in data.get('counts',[])]
    record=dict(phase='mps_summary', source=str(logical_path(source)),source_sha256=content_digest(source)['sha256'],
        source_storage=storage_record(source),
        rows=rows,counts=counts,execution_status='complete' if rows and all(r['execution_status']=='complete' for r in rows) and all(c['status'] in ('complete','skipped') for c in counts) else 'failed',
        interpretation='Direct MPS marginals; counts mapping checks are not rare-response precision or AE. Individual runs are not pooled.')
    out=output_directory(outdir,'mps_summary')
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'statistics.csv').open('x',newline='',encoding='utf-8') as f:
        columns=['bond','p_D','reference_D','relative_error','execution_status','rare_accuracy_status','pilot_accuracy_status','strict_status']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(outdir=str(out),execution_status=record['execution_status'],rows=[{k:r.get(k) for k in ('bond','rare_accuracy_status','pilot_accuracy_status','strict_status')} for r in rows]),ensure_ascii=False),flush=True)
    return 0 if record['execution_status']=='complete' else 3


def _cmd_request(action, explicit_action):
    from run_request import COMMON, APPLICABLE
    if os.environ.get('VRQO_RUN_CMD') != '1': return {}
    if explicit_action and action != os.environ.get('TASK'):
        print('Explicit ACTION selects its own parameters; the run.cmd preset is not applied.',flush=True)
        return {}
    values={k:os.environ[k.upper()] for k in COMMON if os.environ.get(k.upper(),'')!=''}
    if values.get('suite') and action not in APPLICABLE['suite']:
        raise ValueError('Selected TASK cannot use SUITE. Clear SUITE/CASE_ID and select CONFIG explicitly.')
    unused={k:v for k,v in values.items() if k in APPLICABLE and action not in APPLICABLE[k]}
    if unused and os.environ.get('DEBUG_RUNTIME')=='1': print('CMD fields not applicable to selected task (not forwarded): '+json.dumps(unused),flush=True)
    return {k:v for k,v in values.items() if k not in unused}



def _cmd_runtime_flags(action, explicit_action, rest):
    """Explicit preset values reach Python; CLI options retain precedence."""
    if action=='mps_batch' or os.environ.get('VRQO_RUN_CMD') != '1' or (explicit_action and action != os.environ.get('TASK')):
        return []
    supplied={word.partition('=')[0] for word in rest}
    flags=[]
    for name, option in (('AER_HEARTBEAT_SEC','--aer-heartbeat-sec'),
                         ('RESOURCE_SAMPLE_SEC','--resource-sample-sec'),('POLL_SEC','--poll-sec')):
        if os.environ.get(name,'') and option not in supplied:
            flags += [option,os.environ[name]]
    from run_request import _boolean
    if os.environ.get('DEBUG_RUNTIME','') and not supplied & {'--debug-runtime','--no-debug-runtime'}:
        if _boolean(os.environ['DEBUG_RUNTIME'],'DEBUG_RUNTIME'):flags.append('--debug-runtime')
    if action=='test':
        for name,option in (('TEST_MEASURE_X','--test-measure-x'),('TEST_MEASURE_DIR','--test-measure-dir'),
                            ('TEST_MEASURE_WORK','--debug-measure-work'),('AER_SAFE_BASIS','--aer-safe-basis')):
            if os.environ.get(name,'') and option not in supplied and _boolean(os.environ[name],name):flags.append(option)
        if os.environ.get('SHOT_CHUNK','') and '--aer-shot-chunk' not in supplied:
            flags += ['--aer-shot-chunk',os.environ['SHOT_CHUNK']]
    return flags


def _shard_request(action, rest, cmd, *, preset):
    """Translate the existing science CLI to the existing strict shard runner."""
    from main import build_argparser, _resolve_request, REQUEST_FIELDS
    selector=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    selector.add_argument('--shard-mode',choices=('0','1'),default=os.environ.get('SHARD_MODE','0') if preset else '0')
    selector.add_argument('--num-shards',type=int,default=int(os.environ.get('NUM_SHARDS','1')) if preset else 1)
    selector.add_argument('--max-parallel',type=int,default=int(os.environ.get('MAX_PARALLEL','1')) if preset else 1)
    selector.add_argument('--shot-chunk',type=int)
    selection,science=selector.parse_known_args(rest)
    if selection.shard_mode=='0':
        # --num-shards was already a public main.py option; preserve it in single mode.
        unchanged=[];i=0
        while i<len(rest):
            if rest[i].partition('=')[0]=='--shard-mode':i+=1 if '=' in rest[i] else 2
            else:unchanged.append(rest[i]);i+=1
        return None,unchanged
    parser=build_argparser()
    specified={parser._option_string_actions[word.partition('=')[0]].dest for word in science
               if word.partition('=')[0] in parser._option_string_actions}
    # Render only CMD fields not explicitly overridden, using the authoritative parser.
    by_dest={}
    for option in parser._actions:by_dest.setdefault(option.dest,option)
    defaults=[]
    from run_request import _boolean
    for attr,key in REQUEST_FIELDS.items():
        if attr=='phase' or attr in specified or key not in cmd:continue
        option=by_dest[attr].option_strings[0];value=cmd[key]
        if isinstance(by_dest[attr],argparse._StoreTrueAction):
            if _boolean(value,key):defaults.append(option)
        else:defaults += [option,str(value)]
    science=[*defaults,*science]
    args=parser.parse_args(['--phase',action,*science])
    _resolve_request(args,['--phase',action,*science])
    chunk=args.aer_shot_chunk if selection.shot_chunk is None else selection.shot_chunk
    managed={'steps','shots','histories','seed','outdir','dry_run','aer_shot_chunk'}
    extra=[];i=0
    while i<len(science):
        token=science[i];option=parser._option_string_actions[token.partition('=')[0]]
        width=1 if '=' in token or option.nargs==0 else 2
        if option.dest not in managed:extra.extend(science[i:i+width])
        i+=width
    if action=='test':extra += ['--aer-shot-chunk',str(chunk)]
    forwarded=['--phase',action,'--steps',str(args.steps),'--num-shards',str(selection.num_shards),
        '--max-parallel',str(selection.max_parallel),'--seed0',str(args.seed),'--outroot',args.outdir,
        '--shot-chunk',str(chunk if action=='test' else 0),
        '--total-shots' if action=='test' else '--histories-per-shard',str(args.shots if action=='test' else args.histories),
        '--extra-json',json.dumps(extra)]
    if args.dry_run:forwarded.append('--dry-run')
    return forwarded,science


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    actions = ('check-env', 'validate', 'reference', 'screen', 'mps_probe', 'mps_batch', 'mc', 'test', 'circuit', 'iqae', 'mlae', 'shards', 'pilot', 'summarize', 'legacy-reference', 'legacy-validate-env')
    if (not args and os.environ.get('VRQO_RUN_CMD')!='1') or (args and args[0] in ('--help', '-h')):
        print(__doc__)
        print('Actions:', ', '.join(actions))
        print('Use ACTION --help for existing parameters. Validation has no default time limit.')
        return 0
    explicit_action=bool(args and not args[0].startswith('-'))
    action, rest = (args[0],args[1:]) if explicit_action else (os.environ.get('TASK','mc'),args)
    if action not in actions:
        print('Unknown action: ' + action, file=sys.stderr)
        return 2
    if action == 'check-env':
        return check_environment(rest)
    require_qc()
    if action == 'legacy-reference':
        return reference(rest)
    if action == 'summarize':
        if os.environ.get('VRQO_RUN_CMD')=='1' and action==os.environ.get('TASK'):
            defaults=[]
            for name in ('input','outdir'):
                if os.environ.get(name.upper()): defaults += ['--'+name,os.environ[name.upper()]]
            rest=[*defaults,*rest]
        return summarize(rest)
    if action == 'legacy-validate-env':
        reject_action_override(rest, '--task')
        flags = [] if '--dry-run' in rest or '--execute' in rest else ['--execute']
        return invoke('cloud_pilot.py', ['--task', 'env', *flags, *rest])
    if action == 'pilot':
        flags = [] if '--dry-run' in rest or '--execute' in rest else ['--execute']
        return invoke('cloud_pilot.py', [*flags, *rest])
    if action == 'shards':
        return invoke('shard_runner.py', rest)
    reject_action_override(rest, '--phase')
    reject_action_override(rest, '--task')
    cmd=_cmd_request(action,explicit_action)
    specified={word.partition('=')[0] for word in rest}
    if '--suite' in specified:cmd.pop('config',None)
    if '--config' in specified:
        cmd.pop('suite',None);cmd.pop('case_id',None)
    os.environ['VRQO_CMD_REQUEST_JSON']=json.dumps(cmd)
    rest=[*_cmd_runtime_flags(action,explicit_action,rest),*rest]
    if action in ('test','mc'):
        forwarded,rest=_shard_request(action,rest,cmd,
            preset=os.environ.get('VRQO_RUN_CMD')=='1' and action==os.environ.get('TASK'))
        if forwarded is not None:
            # Every child receives its explicit allocated count and output path.
            os.environ['VRQO_CMD_REQUEST_JSON']='{}'
            return invoke('shard_runner.py',forwarded)
    return invoke('main.py', ['--phase', action, *rest])


if __name__ == '__main__':
    raise SystemExit(main())
