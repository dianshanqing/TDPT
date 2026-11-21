"""Bounded actual-gate checks and instrumented Aer marginal-probability probes.

No wall-clock cutoff, retries, AE runs or truth-based state preparation. Sparse
state-count budgets and finite suites bound the scientific scope instead.
"""
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
import time
import traceback

from models import load_document, load_model, resolve_path
from runtime_identity import (circuit_fingerprint, make_identity, normalize,
                              validate_output_path, write_json_new)
from supervised import publish_progress

TOL = 2e-11
ORDER = ['survive', 'detect', 'absorb', 'boundary']


def _peak_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, 'peak_wset', info.rss) / 1024**2
    except ImportError:
        return None


def _output(outdir):
    p = validate_output_path(resolve_path(outdir))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _suite(path):
    return load_document(path)


def _cpu(cpu_budget):
    cpu_budget = _positive_integer(cpu_budget, 'cpu_budget')
    # Deliberately serial small probes; total workers * threads = 1.
    return {'requested_total_budget': int(cpu_budget), 'workers': 1, 'threads_per_worker': 1}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer (not bool or float)')
    return int(value)


def _validated_cases(spec):
    cases = spec.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('suite cases must be a nonempty list')
    for case in cases:
        _positive_integer(case['steps'], 'case steps')
    return cases


def _probe_settings(spec):
    bonds = spec.get('bond_dimensions')
    thresholds = spec.get('truncation_thresholds')
    if not isinstance(bonds, list) or not bonds:
        raise ValueError('bond_dimensions must be a nonempty list')
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError('truncation_thresholds must be a nonempty list')
    bonds = [_positive_integer(b, 'bond dimension') for b in bonds]
    for value in thresholds:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError('truncation thresholds must be finite nonnegative numbers')
    return bonds, thresholds, _positive_integer(spec.get('shots',256), 'shots')


def _sparse_crosscheck(circuit, reference_probabilities, enabled=True, max_states=60000):
    if not isinstance(enabled, bool):
        raise ValueError('sparse_crosscheck must be a JSON boolean')
    if not enabled:
        return {'sparse_crosscheck':False, 'sparse_actual':None, 'sparse_runtime':None,
                'sparse_passed':None, 'sparse_status_max_error':None,
                'sparse_skip_reason':'Explicit suite/case sparse_crosscheck=false for a main-candidate probe; independent DP remains active. No sparse accuracy claim.'}
    from reference_engine import evolve, observe
    state, runtime = evolve(circuit, max_states=max_states)
    actual = observe(circuit, state)
    error = max(abs(a-b) for a,b in zip(actual['probabilities'], reference_probabilities))
    return {'sparse_crosscheck':True, 'sparse_actual':actual, 'sparse_runtime':runtime,
            'sparse_passed':error < TOL and actual['work_nonzero'] < TOL and abs(actual['norm']-1) < TOL,
            'sparse_status_max_error':error, 'sparse_skip_reason':None}


def _error(got, want):
    return max((abs(got.get(k, 0) - want.get(k, 0)) for k in got.keys() | want.keys()), default=0.)


def _identity(cfg, steps, profile, circuit, backend, phase, original_fingerprint=None,
              compiled_fingerprint=None, diagnostic_fingerprint=None):
    from reference_engine import registers
    from unitary_circuit import resolve_kernel_profile
    compiled = phase == 'mps_probe' or compiled_fingerprint is not None
    extra={'steps': steps, 'original_A_fingerprint': original_fingerprint or circuit_fingerprint(circuit),
           'sparse_engine_sha256': hashlib.sha256(resolve_path('reference_engine.py').read_bytes()).hexdigest()}
    if compiled_fingerprint is not None:extra['compiled_A_fingerprint']=compiled_fingerprint
    if diagnostic_fingerprint is not None:extra['diagnostic_fingerprint']=diagnostic_fingerprint
    return make_identity(config=cfg, profile=asdict(resolve_kernel_profile(kernel_profile=profile)),
        compile_options=({'basis_gates':['u','cx'], 'optimization_level':0, 'seed_transpiler':917, 'num_processes':1}
                         if compiled else {'transpiled':False, 'evolution':'original actual gate definitions'}),
        measurement={'status_order': ORDER, 'register_map': registers(circuit)},
        backend=backend, phase=phase,
        extra=extra)


def _joint_rows(joint):
    return [list(k) + [v] for k, v in sorted(joint.items())]


def _mc(cfg, steps, histories, expected):
    from mc import run_mc
    started = time.perf_counter()
    result = run_mc(cfg, steps, histories, seed=20261030)
    probabilities = [result.p_survive, result.p_detect, result.p_absorb, result.p_boundary]
    # An explicitly loose diagnostic comparison, not a claimed precision target.
    limits = [6 * math.sqrt(p * (1-p) / histories) + 6 / histories for p in expected]
    return {'histories': histories, 'seed': 20261030, 'counts': result.counts,
        'probabilities': probabilities, 'reference': expected,
        'diagnostic_limits': limits, 'diagnostic_pass': all(abs(a-b) <= lim for a,b,lim in zip(probabilities, expected, limits)),
        'seconds': time.perf_counter()-started}


def _sparse_case(cfg, steps, profile, max_states):
    from reference_solver import reference
    from reference_engine import build_traced, evolve, marginal, observe, registers
    started = time.perf_counter()
    circuit, info, points = build_traced(cfg, steps, profile)
    build_seconds = time.perf_counter()-started
    def observer(qc, state, point):
        got = marginal(qc, state, ['status', 'x', 'y', 'z', 'dir', 'g'])
        wanted = reference(cfg, int(point['step'])+1, retain_joint=True)['joint']
        return dict(observe(qc, state), max_joint_error=_error(got, wanted))
    state, runtime = evolve(circuit, max_states=max_states, checkpoints=points, observer=observer)
    got = marginal(circuit, state, ['status', 'x', 'y', 'z', 'dir', 'g'])
    wanted = reference(cfg, steps, retain_joint=True)['joint']
    observed = observe(circuit, state)
    max_error = _error(got, wanted)
    per_step_ok = all(row['observables']['max_joint_error'] < TOL
                      and row['observables']['work_nonzero'] < TOL for row in runtime['checkpoints'])
    row = {'status': 'complete', 'model': cfg.model, 'steps': steps, 'profile': profile,
        'identity': _identity(cfg, steps, profile, circuit, {'method': 'sparse_actual_gates', 'max_states': max_states}, 'validate'),
        'qubits': circuit.num_qubits, 'register_map': registers(circuit), 'resource_estimate': info,
        'top_level_gates': len(circuit.data), 'depth': circuit.depth(), 'build_seconds': build_seconds,
        'physical_joint_columns': ['status','x','y','z','dir','g','probability'],
        'actual_joint': _joint_rows(got), 'reference_joint': _joint_rows(wanted),
        'max_joint_error': max_error, 'absolute_tolerance': TOL, 'observed': observed,
        'runtime': runtime, 'peak_process_memory_mb': _peak_mb(),
        'passed': max_error < TOL and observed['work_nonzero'] < TOL and abs(observed['norm']-1) < TOL and per_step_ok}
    row['total_build_evolution_identity_seconds'] = time.perf_counter()-started
    return row


def _inverse_and_reflection(cfg, max_states):
    from reference_engine import evolve
    from unitary_circuit import build_unitary_circuit_qiskit
    from iqae import OracleSpec, build_grover_operator_for_spec
    rows = []
    for profile in ('baseline','compressed','full_history'):
        circuit, _ = build_unitary_circuit_qiskit(cfg, 1, kernel_profile=profile)
        state, _ = evolve(circuit, max_states=max_states)
        back, runtime = evolve(circuit.inverse(), state, max_states=max_states)
        regs = {r.name:r for r in circuit.qregs}
        q, _, _ = build_grover_operator_for_spec(A_circuit=circuit, status_qubits=regs['status'],
            g_qubits=regs['g'], spec=OracleSpec(good='detect'))
        amplitude_error = abs(back.get(0, 0)-1)
        tail = math.fsum(abs(a)**2 for k,a in back.items() if k)
        full = list(q.reflection_qubits) == list(range(q.num_qubits))
        rows.append({'profile': profile, 'inverse_amplitude_error': amplitude_error,
            'inverse_tail_probability': tail, 'all_registers_reflected': full,
            'reflection_qubits': list(q.reflection_qubits), 'runtime': runtime,
            'passed': amplitude_error < TOL and tail < 2e-20 and full})
    return rows


def run_validation(suite_path, outdir, cpu_budget=1):
    """Run finite profiles and MC cross-checks; preserve individual failures."""
    spec = _suite(suite_path)
    if spec.get('kind') == 'numerical_readout_audit':
        from mps_numerical import run_numerical_audit
        _cpu(cpu_budget)
        if os.environ.get('VRQO_DEADLINE_MONOTONIC'):
            raise ValueError('Inherited deadline is incompatible with numerical audit')
        return run_numerical_audit(spec,outdir,cpu_budget)
    if spec.get('kind') == 'svd_calibration':
        from rare_precision import run_svd_calibration
        _cpu(cpu_budget)
        if os.environ.get('VRQO_DEADLINE_MONOTONIC'):
            raise ValueError('Inherited deadline is incompatible with unlimited calibration')
        return run_svd_calibration(spec,outdir,cpu_budget)
    if spec.get('kind') == 'precision_audit':
        from rare_precision import run_precision_audit
        _cpu(cpu_budget)
        return run_precision_audit(spec,outdir,cpu_budget)
    if spec.get('kind') == 'bounded_sparse_audit':
        from rare_precision import run_bounded_sparse_audit
        _cpu(cpu_budget);_validated_cases(spec)
        return run_bounded_sparse_audit(spec,outdir,cpu_budget)
    cases = _validated_cases(spec)
    output = _output(outdir); cpu = _cpu(cpu_budget)
    if float(spec.get('absolute_tolerance', TOL)) != TOL:
        raise ValueError('Existing absolute probability tolerance is frozen at 2e-11')
    inherited = os.environ.get('VRQO_DEADLINE_MONOTONIC')
    if inherited:
        raise ValueError('VRQO_DEADLINE_MONOTONIC is set; unlimited validation refuses an inherited deadline')
    from reference_solver import reference
    all_rows = []; mc_rows = []; started = time.perf_counter()
    for case in cases:
        cfg = load_model(case['config']); steps = _positive_integer(case['steps'], 'case steps')
        for profile in case.get('profiles', spec.get('profiles', ['baseline'])):
            stem = case['id'] + '_' + profile
            print(json.dumps({'stage':'sparse_start', 'case':stem, 'steps':steps}), flush=True)
            try:
                row = _sparse_case(cfg, steps, profile, int(spec.get('max_sparse_states', 60000)))
            except Exception as exc:
                row = {'status':'failed', 'case':case, 'profile':profile, 'error':repr(exc), 'traceback':traceback.format_exc(), 'passed':False}
            write_json_new(output/(stem+'.json'), row)
            all_rows.append({'case':stem, 'status':row['status'], 'passed':row['passed'],
                'max_joint_error':row.get('max_joint_error'), 'seconds':row.get('runtime',{}).get('seconds')})
            print(json.dumps(all_rows[-1]), flush=True)
        mc_row = dict(case=case['id'], **_mc(cfg, steps, int(spec.get('mc_histories',4000)), reference(cfg,steps)['rows'][-1]['probabilities']))
        mc_rows.append(mc_row)
    checks = _inverse_and_reflection(load_model(spec['cases'][0]['config']), int(spec.get('max_sparse_states',60000)))
    report = {'status':'complete', 'cpu':cpu, 'rows':all_rows, 'mc':mc_rows, 'inverse_and_reflection':checks,
        'absolute_tolerance':TOL, 'elapsed_seconds':time.perf_counter()-started,
        'peak_process_memory_mb':_peak_mb(), 'deadline':None,
        'passed':all(r['passed'] for r in all_rows+checks) and all(r['diagnostic_pass'] for r in mc_rows)}
    write_json_new(output/'validation_summary.json', report)
    lines = ['# Actual gate validation', '', f"Passed: {report['passed']}. Absolute tolerance: {TOL}.",
        'Small instances only. Each step checks the independent status/position/direction/group joint distribution and joint work-zero probability.',
        'Terminal physical-register freezing is checked by continued actual gate evolution through T=5; first detection is T=3 in the three model cases.',
        'Non-power-of-two nx=5, solid no-duct geometry, two material interfaces, pure absorption, vacuum, repeated scattering and boundary priority are included.',
        'Monte Carlo uses 4000 real histories per case and loose 6-sigma diagnostic limits; this does not certify 1% precision.', '',
        '| Case | Passed | Joint maximum error | Seconds |', '|---|---:|---:|---:|']
    lines += [f"| {r['case']} | {r['passed']} | {r['max_joint_error']} | {r['seconds']} |" for r in all_rows]
    (output/'validation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def build_circuit_report(cfg, steps, outdir, profile='baseline', cpu_budget=1):
    from unitary_circuit import build_unitary_circuit_qiskit
    started = time.perf_counter(); circuit, info = build_unitary_circuit_qiskit(cfg, steps, kernel_profile=profile)
    row = {'status':'complete', 'identity':_identity(cfg,steps,profile,circuit,{'method':'build_only'},'circuit'),
        'qubits':circuit.num_qubits, 'top_level_gates':len(circuit.data), 'depth':circuit.depth(),
        'resource_estimate':info, 'seconds':time.perf_counter()-started, 'cpu':_cpu(cpu_budget)}
    write_json_new(_output(outdir)/'circuit_resources.json', row)
    return row


def _probe_seed(value):
    if isinstance(value,bool) or not isinstance(value,Integral) or value < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(value)


def _probe_backend_options(cpu_budget,aer_threads,mps_threads,extras,bond,threshold,seed=20261030):
    budget=_positive_integer(cpu_budget,'cpu_budget')
    aer=_positive_integer(1 if aer_threads is None else aer_threads,'aer_threads')
    mps=_positive_integer(1 if mps_threads is None else mps_threads,'mps_threads')
    if aer*mps>budget:raise ValueError('aer_threads*mps_threads exceeds the total CPU budget')
    allowed={'precision','zero_threshold','chop_threshold','fusion_enable','enable_truncation','mps_log_data','mps_lapack'}
    if set(extras)-allowed:raise ValueError(f'Unsupported diagnostic backend override: {sorted(set(extras)-allowed)}')
    return dict({'method':'matrix_product_state','max_parallel_threads':aer,'max_parallel_experiments':1,
        'max_parallel_shots':1,'mps_omp_threads':mps,'matrix_product_state_max_bond_dimension':int(bond),
        'matrix_product_state_truncation_threshold':float(threshold),'seed_simulator':_probe_seed(seed),'mps_log_data':True},**extras)


def _probe_accuracy_status(actual, reference, work_upper_bound, *, work_bit_probabilities=None):
    """Separate execution, target response, historical tolerance and validity.

    Raw marginals are never normalized/clipped. Conservation/work anomalies
    require investigation even if D agrees; absolute-tolerance failure alone
    does not request a larger bond or certify/deny relative response precision.
    """
    from rare_precision import compare_rare_probability
    actual_valid=(actual is not None and len(actual)==4 and
                  all(isinstance(x,Real) and math.isfinite(x) and 0<=x<=1 for x in actual))
    reference_valid=(reference is not None and len(reference)==4 and
                     all(isinstance(x,Real) and math.isfinite(x) and 0<=x<=1 for x in reference)
                     and abs(math.fsum(reference)-1.)<TOL)
    finite_actual=(actual is not None and len(actual)==4 and
                   all(isinstance(x,Real) and math.isfinite(x) for x in actual))
    raw_sum=math.fsum(actual) if finite_actual else None
    norm=abs(raw_sum-1.) if raw_sum is not None else None
    errors=([abs(a-b) for a,b in zip(actual,reference)]
            if finite_actual and reference_valid else None)
    comparison=None
    if reference_valid and actual is not None and len(actual)==4:
        try: comparison=compare_rare_probability(actual[1],reference[1])
        except (TypeError,ValueError): pass
    strict=bool(actual_valid and reference_valid and max(errors)<TOL and norm<TOL)
    work_bits_valid=(work_bit_probabilities is None or
                     all(isinstance(x,Real) and math.isfinite(x) and 0<=x<=1
                         for x in work_bit_probabilities))
    work_valid=(isinstance(work_upper_bound,Real) and math.isfinite(work_upper_bound)
                and work_upper_bound>=0 and work_bits_valid)
    work_bound_consistent=(work_bit_probabilities is None or (work_valid and
        abs(math.fsum(work_bit_probabilities)-work_upper_bound)<TOL))
    work_valid=work_valid and work_bound_consistent
    work=bool(work_valid and work_upper_bound<TOL)
    issues=[]
    if not actual_valid:issues.append('missing_nonfinite_negative_or_out_of_range_probability')
    if not reference_valid:issues.append('invalid_reference')
    if norm is not None and norm>=TOL:issues.append('conservation_residual_requires_target_impact_assessment')
    if not work_bits_valid:issues.append('nonfinite_negative_or_out_of_range_work_bit_probability')
    if not work_bound_consistent:issues.append('reported_work_union_bound_inconsistent_with_raw_bits')
    if not work_valid:issues.append('missing_or_invalid_work_probability')
    elif not work:issues.append('work_leakage_requires_circuit_validity_assessment')
    validity=('failed' if not actual_valid or not reference_valid else 'unresolved' if issues else 'passed')
    def status(value):
        return 'not_applicable' if value is None else ('passed' if value else 'failed')
    return {'execution_status':'passed',
        'rare_accuracy_status':status(comparison['passes_formal_1pct']) if comparison else 'failed',
        'pilot_accuracy_status':status(comparison['passes_pilot_5pct']) if comparison else 'failed',
        'strict_status':status(strict),'work_accuracy_status':status(work),
        'passed_absolute_tolerance':strict,'passed_status_absolute_tolerance':strict,
        'passed_work_upper_bound':work,'raw_probability_sum':raw_sum,'normalization_residual':norm,
        'absolute_errors':errors,'zero_reference_exact_match':comparison['passes_zero_reference'] if comparison else None,
        'numerical_validity_status':validity,'unresolved_numerical_issues':issues,
        'scientific_certification_passed':bool(validity=='passed' and comparison and
            (comparison['passes_formal_1pct'] or comparison['passes_zero_reference'])),
        'statistical_accuracy_status':'not_applicable_direct_probability',
        'physical_discretization_status':'not_evaluated',
        'certification_scope':'fixed-model direct D response only; excludes physical discretization, IQAE and strict full-distribution certification',
        'strict_tolerance_role':'historical diagnostic; not an automatic bond escalation trigger'}


def _diagnostic_json_values(value):
    """Retain nonfinite raw evidence in strict JSON without inventing zeros."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float':repr(value)}
    if isinstance(value, dict):
        return {key:_diagnostic_json_values(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_values(item) for item in value]
    return value


def _aggregate_probe_status(rows, key):
    states=[row.get(key,'not_evaluated') for row in rows]
    for state in ('failed','unresolved','not_evaluated','passed'):
        if state in states:return state
    return 'not_applicable'


def _probe_summary_row(row, result_path):
    """Keep scientific fields inline and reference the authoritative verbose log.

    The per-setting JSON has already been written and retains the untouched Aer
    metadata. Hash in chunks so a large log does not require another full copy.
    Paths are relative to mps_probability_probe.json in the same run directory.
    """
    metadata=row.get('metadata',{})
    if 'MPS_log_data' not in metadata:
        return dict(row)
    from result_storage import logical_path, storage_record
    result_path=Path(result_path)
    stored=storage_record(result_path)
    stored['path']=Path(stored['path']).name
    stored['logical_path']=Path(stored['logical_path']).name
    log=metadata['MPS_log_data'];log_hash=hashlib.sha256();log_bytes=0
    for start in range(0,len(log),65536):
        chunk=log[start:start+65536].encode('utf-8')
        log_hash.update(chunk);log_bytes+=len(chunk)
    return {**row,'metadata':{key:value for key,value in metadata.items() if key!='MPS_log_data'},
        'metadata_log_references':{'MPS_log_data':{
            'path':logical_path(result_path).name,'json_pointer':'/metadata/MPS_log_data',
            'file_sha256':stored['original_sha256'],'file_bytes':stored['original_bytes'],
            'storage':stored,
            'log_sha256':log_hash.hexdigest(),'log_utf8_bytes':log_bytes}}}


def run_mps_probe(suite_path, outdir, cpu_budget=1, *, aer_threads=None, mps_threads=None, blas_threads=1, backend_options=None, counts_check=None, seed=None):
    """Save two status probabilities before measurement, without full-state save."""
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_probabilities
    from reference_solver import reference
    from reference_engine import registers
    from unitary_circuit import build_unitary_circuit_qiskit
    from runtime_identity import apply_backend_options
    from numerical_runtime import configure_blas_threads, numerical_libraries
    from rare_precision import compare_rare_probability, compiled_sparse_probe
    spec = _suite(suite_path)
    # None means unspecified; explicit arguments override the suite, including True.
    counts_check=spec.get('counts_check',True) if counts_check is None else counts_check
    if not isinstance(counts_check,bool):raise ValueError('counts_check must be a boolean')
    seed=_probe_seed(spec.get('seed',20261030) if seed is None else seed)
    cases = _validated_cases(spec)
    bonds, thresholds, shots = _probe_settings(spec)
    blas_threads=_positive_integer(blas_threads,'blas_threads')
    if (aer_threads or 1)*(mps_threads or 1)*blas_threads>cpu_budget:
        raise ValueError('Aer * MPS * BLAS threads exceeds CPU budget')
    blas_application=configure_blas_threads(blas_threads)
    output = _output(outdir); cpu = _cpu(cpu_budget)
    cpu.update(threads_per_worker=None,aer_threads=1 if aer_threads is None else aer_threads,
               mps_threads=1 if mps_threads is None else mps_threads,blas_threads=blas_threads,
               blas_application=blas_application)
    if os.environ.get('VRQO_DEADLINE_MONOTONIC'):
        raise ValueError('Inherited deadline is incompatible with unlimited probe')
    rows = []; resources = []; count_rows = []; started = time.perf_counter()
    # Four deterministic 2-qubit basis states validate saves, measurement and decoding.
    # This small interface check remains active when the costly main-A counts are off.
    publish_progress('mapping_check')
    tiny_cases=[]
    for value in range(4):
        tiny=QuantumCircuit(2,2)
        for bit in range(2):
            if value & (1 << bit):tiny.x(bit)
        tiny.save_probabilities([0,1],label='status');tiny.measure([0,1],[0,1])
        tiny_cases.append(tiny)
    tiny_result=AerSimulator(method='matrix_product_state',max_parallel_threads=1,
        max_parallel_experiments=1,max_parallel_shots=1,mps_omp_threads=1,
        seed_simulator=seed).run(tiny_cases,shots=1).result()
    mapping_rows=[]
    if not tiny_result.success:raise RuntimeError(str(tiny_result.status))
    for value in range(4):
        saved=tiny_result.data(value)['status'].tolist();raw=tiny_result.get_counts(value)
        expected=[float(i==value) for i in range(4)]
        passed=max(abs(a-b) for a,b in zip(saved,expected)) < TOL and raw == {format(value,'02b'):1}
        mapping_rows.append({'basis_index':value,'probabilities':saved,'raw_counts':raw,'passed':passed})
        if not passed:raise AssertionError('Aer little-endian save/measurement mapping failed')
    check=mapping_rows[1]['probabilities']
    for case in cases:
        cfg=load_model(case['config']); steps=_positive_integer(case['steps'], 'case steps'); profile=case.get('profile','baseline')
        publish_progress('build',case=case['id'],steps=steps,profile=profile)
        tick=time.perf_counter();cpu_tick=time.process_time(); circuit,info=build_unitary_circuit_qiskit(cfg,steps,kernel_profile=profile)
        build_seconds=time.perf_counter()-tick
        build_cpu_seconds=time.process_time()-cpu_tick
        ref=reference(cfg,steps)['rows'][-1]['probabilities']
        sparse_check = _sparse_crosscheck(circuit, ref,
            enabled=case.get('sparse_crosscheck', spec.get('sparse_crosscheck', True)))
        tick=time.perf_counter();original_fingerprint=circuit_fingerprint(circuit)
        original_fingerprint_seconds=time.perf_counter()-tick
        publish_progress('compile',case=case['id'],build_seconds=build_seconds)
        tick=time.perf_counter();cpu_tick=time.process_time()
        # Fixed standard basis gives a reproducible compiled A for every bond.
        compiled=transpile(circuit,basis_gates=['u','cx'],optimization_level=0,seed_transpiler=917,num_processes=1)
        compile_seconds=time.perf_counter()-tick
        compile_cpu_seconds=time.process_time()-cpu_tick
        compiled_crosscheck = (compiled_sparse_probe(compiled,ref,
            _positive_integer(spec.get('max_compiled_sparse_states',60000),'max_compiled_sparse_states'))
            if case.get('compiled_sparse_crosscheck',spec.get('compiled_sparse_crosscheck',False)) else None)
        regs=registers(compiled); instrumented=compiled.copy()
        instrumented.save_probabilities(regs['status'],label='status_probability')
        for i,bit in enumerate(regs['work']):
            instrumented.save_probabilities([bit],label=f'work_{i}')
        tick=time.perf_counter();diagnostic_fingerprint=circuit_fingerprint(instrumented)
        diagnostic_fingerprint_seconds=time.perf_counter()-tick
        tick=time.perf_counter();compiled_fingerprint=circuit_fingerprint(compiled)
        compiled_fingerprint_seconds=time.perf_counter()-tick
        base={'case':case,'config':asdict(cfg),'original_A_fingerprint':original_fingerprint,
            'compiled_A_fingerprint':compiled_fingerprint,'diagnostic_fingerprint':diagnostic_fingerprint,
            'original_fingerprint_seconds':original_fingerprint_seconds,'compiled_fingerprint_seconds':compiled_fingerprint_seconds,
            'diagnostic_fingerprint_seconds':diagnostic_fingerprint_seconds,
            'register_map':regs,'reference':ref,**sparse_check,'compiled_sparse':compiled_crosscheck,
            'qubits':circuit.num_qubits,'top_level_gates':len(circuit.data),'compiled_gates':len(compiled.data),
            'compiled_two_qubit_gates':sum(len(i.qubits)==2 for i in compiled.data),'compiled_depth':compiled.depth(),
            'build_seconds':build_seconds,'compile_seconds':compile_seconds,'save_qubit_count':2,
            'build_process_cpu_seconds':build_cpu_seconds,'compile_process_cpu_seconds':compile_cpu_seconds}
        settings=([(b,t) for b in bonds for t in thresholds] if spec.get('full_bond_threshold_grid',False)
            else [(b,thresholds[0]) for b in bonds]+[(bonds[-1],t) for t in thresholds[1:]])
        extras=dict(spec.get('backend_options',{}));extras.update(case.get('backend_options',{}));extras.update(backend_options or {})
        for bond,threshold in settings:
            label=f"{case['id']}_bond{bond}_trunc{threshold:g}"
            print(json.dumps({'stage':'mps_start','case':label,'qubits':circuit.num_qubits,'compiled_gates':len(compiled.data)}),flush=True)
            options=_probe_backend_options(cpu_budget,aer_threads,mps_threads,extras,bond,threshold,seed=seed)
            try:
                sim=AerSimulator();actual_backend=apply_backend_options(sim,options)
                libraries_before=numerical_libraries()
                blas_application=configure_blas_threads(blas_threads)
                publish_progress('mps_evolution_and_probability_read',case=case['id'],bond=bond,
                    backend_options=actual_backend, timing_scope='combined backend call')
                tick=time.perf_counter();cpu_tick=time.process_time(); result=sim.run(instrumented,shots=1).result()
                evolution_and_read_seconds=time.perf_counter()-tick
                evolution_cpu_seconds=time.process_time()-cpu_tick
                if not result.success: raise RuntimeError(str(result.status))
                data=result.data(); actual=data['status_probability'].tolist()
                work=[float(data[f'work_{i}'][1]) for i in range(len(regs['work']))]
                errors=[abs(a-b) for a,b in zip(actual,ref)]
                relative=errors[1]/ref[1] if ref[1] else None
                row=dict(base,status='complete',bond=bond,truncation_threshold=threshold,probabilities=actual,
                    absolute_errors=errors,p_D_relative_error=relative,passes_1_percent=(relative<=.01 if relative is not None else None),
                    passes_5_percent=(relative<=.05 if relative is not None else None),
                    work_bit_probabilities=work,work_any_nonzero_union_upper_bound=sum(work),
                    evolution_and_observable_read_seconds=evolution_and_read_seconds,
                    evolution_and_observable_process_cpu_seconds=evolution_cpu_seconds,
                    evolution_and_observable_average_process_cores=evolution_cpu_seconds/evolution_and_read_seconds,
                    cpu_timing_scope='worker process including native threads; whole-tree CPU is in supervision_result.json',
                    blas_threads=blas_threads,blas_application=blas_application,
                    numerical_libraries_before=libraries_before,numerical_libraries_after=numerical_libraries(),
                    separate_evolution_and_read_timing='not exposed by this Aer execution API',
                    peak_process_memory_mb=_peak_mb(),backend_options=options,backend=actual_backend,
                    rare_comparison=(compare_rare_probability(actual[1],ref[1])
                        if math.isfinite(actual[1]) and 0<=actual[1]<=1 else None),metadata=normalize(result.results[0].metadata),
                    identity=_identity(cfg,steps,profile,circuit,actual_backend,'mps_probe',original_fingerprint,
                                       compiled_fingerprint,diagnostic_fingerprint))
                row.update(_probe_accuracy_status(actual,ref,sum(work),work_bit_probabilities=work))
            except Exception as exc:
                row=dict(base,status='failed',execution_status='failed',rare_accuracy_status='not_evaluated',
                    pilot_accuracy_status='not_evaluated',strict_status='not_evaluated',
                    bond=bond,truncation_threshold=threshold,error=repr(exc),traceback=traceback.format_exc())
            publish_progress('write',case=case['id'],artifact=label+'.json')
            row=_diagnostic_json_values(row)
            result_path=write_json_new(output/(label+'.json'),row)
            rows.append(_probe_summary_row(row,result_path))
            print(json.dumps({'case':label,'status':row['status'],'execution_status':row['execution_status'],
                'rare_accuracy_status':row['rare_accuracy_status'],'pilot_accuracy_status':row['pilot_accuracy_status'],
                'strict_status':row['strict_status'],'error':row.get('p_D_relative_error'),
                'seconds':row.get('evolution_and_observable_read_seconds')}),flush=True)
        # Counts are sampled from actual compiled A plus terminal measurement;
        # no save/reset is present in the hardware-unitary circuit or its inverse.
        count_row={'status':'skipped','executed':False,'case':case,'shots':None,'raw_counts':None,
            'decoded_status_counts':None,'sample_probabilities':None,'sampling_seconds':None,
            'measured_fingerprint':None,'original_A_fingerprint':original_fingerprint,
            'reason':'counts_check=false; the main measured A was not constructed or executed',
            'purpose':'No main-case finite-count measurement validation was performed'}
        if counts_check:
            publish_progress('counts',case=case['id'],shots_total=shots,backend_options=options)
            try:
                measured=compiled.copy();cr=ClassicalRegister(2,'status_readout');measured.add_register(cr)
                measured.measure([measured.qubits[b] for b in regs['status']],list(cr))
                tick=time.perf_counter(); counts_result=AerSimulator(**options).run(measured,shots=shots).result()
                if not counts_result.success: raise RuntimeError(str(counts_result.status))
                counts=counts_result.get_counts();decoded=[sum(v for k,v in counts.items() if int(k.replace(' ',''),2)==i) for i in range(4)]
                if sum(decoded) != shots:
                    raise AssertionError('Decoded status counts do not equal the requested shots')
                count_row={'status':'complete','executed':True,'case':case,'shots':sum(decoded),'raw_counts':counts,'decoded_status_counts':decoded,
                    'status_order':ORDER,'sampling_seconds':time.perf_counter()-tick,'backend_options':options,
                    'measured_fingerprint':circuit_fingerprint(measured),'original_A_fingerprint':original_fingerprint,
                    'sample_probabilities':[n/sum(decoded) for n in decoded], 'reference':ref,
                    'purpose':'measurement ordering and finite-count interface; not certification of MPS systematic error'}
            except Exception as exc:
                count_row={'status':'failed','executed':True,'case':case,'error':repr(exc),'traceback':traceback.format_exc()}
        else:
            publish_progress('counts_skipped',case=case['id'],reason=count_row['reason'])
        publish_progress('write',case=case['id'],artifact=case['id']+'_counts.json')
        write_json_new(output/(case['id']+'_counts.json'),count_row)
        count_rows.append(count_row)
        resources.append({k:base[k] for k in ('qubits','top_level_gates','compiled_gates','compiled_two_qubit_gates','compiled_depth','build_seconds','compile_seconds')})
    execution_passed=all(r['status']=='complete' for r in rows) and all(r['status'] in ('complete','skipped') for r in count_rows)
    report={'status':'complete','execution_status':'passed' if execution_passed else 'failed',
        'cpu':cpu,'deadline':None,'status_order':ORDER,'little_endian_save_check':check,
        'little_endian_measurement_check':{'passed':True,'rows':mapping_rows},'counts_check':counts_check,'seed':seed,
        'rows':rows,'counts':count_rows,'resources':resources,'elapsed_seconds':time.perf_counter()-started,
        'passed':execution_passed,
        **{key:_aggregate_probe_status(rows,key) for key in ('rare_accuracy_status','pilot_accuracy_status','strict_status','work_accuracy_status','numerical_validity_status')},
        'scientific_certification_passed':all(r.get('scientific_certification_passed',False) for r in rows),
        'sparse_accuracy_status':('failed' if any(r.get('sparse_passed') is False for r in rows)
            else 'passed' if any(r.get('sparse_passed') is True for r in rows) else 'not_applicable'),
        'all_pass_absolute_tolerance':all(r.get('passed_absolute_tolerance',False) for r in rows),
        'interpretation':'Small simulator diagnostics only. Direct marginals remove shot error, not finite bond/truncation/floating error; no hardware advantage claim.'}
    publish_progress('write',artifact='mps_probability_probe.json')
    write_json_new(output/'mps_probability_probe.json',report)
    with (output/'resource_summary.csv').open('w',newline='',encoding='utf-8') as f:
        columns=['bond','truncation_threshold','status','qubits','compiled_gates','compiled_two_qubit_gates','compiled_depth','evolution_and_observable_read_seconds','peak_process_memory_mb','p_D_relative_error']
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    publish_progress('mps_probe_complete',execution_status=report['execution_status'],
        rare_accuracy_status=report['rare_accuracy_status'],strict_status=report['strict_status'])
    return report
