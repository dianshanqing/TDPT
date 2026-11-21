"""Portable, single-worker actual-gate IQAE supplement for frozen VRQO v2.0.0.

Copy beside the frozen Python files. run.cmd calls --bootstrap-from-cmd with
Conda BASE Python. The quoted QC environment then runs --from-cmd. DRY_RUN=1
checks inputs and prints the protocol without evolving or creating output.
The scientific circuit, sparse operations and estimator are reused unchanged.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROTOCOL = dict(
    model='q_tau19_two_cell', steps=3, profile='baseline', objective='detect',
    seed=2026111701, relative_error=.01, confidence=.95, shots_per_round=512,
    max_shots=20000, max_queries=10000000, max_power=2048, max_states=60000,
    max_actual_primitive_gate_applications=8000000, memory_mib=1536,
    threads=1, norm_work_tolerance=2e-11, wall_deadline_seconds=None,
)
ENV_FIELDS = {
    'SEED': ('seed', int), 'RELATIVE_ERROR': ('relative_error', float),
    'CONFIDENCE': ('confidence', float), 'SHOTS': ('shots_per_round', int),
    'IQAE_MAX_SHOTS': ('max_shots', int),
    'IQAE_MAX_QUERIES': ('max_queries', int),
    'IQAE_MAX_GROVER_POWER': ('max_power', int),
    'IQAE_MAX_SPARSE_STATES': ('max_states', int),
    'IQAE_MAX_ACTUAL_GATES': ('max_actual_primitive_gate_applications', int),
    'MEMORY_MIB': ('memory_mib', int),
}


def protocol_from_env(env):
    p = DEFAULT_PROTOCOL.copy()
    fixed = {'TASK': 'iqae_sparse_complete', 'CONFIG': 'q_tau19_two_cell',
             'STEPS': '3', 'PROFILE': 'baseline', 'SHARD_MODE': '0',
             'CPU_BUDGET': '1', 'AER_THREADS': '1', 'MPS_THREADS': '1',
             'BLAS_THREADS': '1', 'NUM_SHARDS': '1', 'MAX_PARALLEL': '1'}
    for key, expected in fixed.items():
        value = env.get(key, '').strip()
        if value and value != expected:
            raise ValueError(f'{key} must be {expected} for this single-case task')
    for key in ('SUITE', 'CASE_ID', 'IQAE_DIAGNOSTIC', 'IQAE_POWERS',
                'IQAE_COMPARE_TO', 'PERF_PREFIX_GATES', 'PERF_REFERENCE'):
        if env.get(key, '').strip():
            raise ValueError(f'{key} must be empty for complete sparse IQAE')
    for key, (field, cast) in ENV_FIELDS.items():
        if env.get(key, '').strip():
            p[field] = cast(env[key].strip())
    for field in ('seed', 'max_power'):
        if p[field] < 0:
            raise ValueError(f'{field} must be nonnegative')
    for field in ('shots_per_round', 'max_shots', 'max_queries', 'max_states',
                  'max_actual_primitive_gate_applications', 'memory_mib'):
        if p[field] <= 0:
            raise ValueError(f'{field} must be positive')
    for field in ('relative_error', 'confidence'):
        if not math.isfinite(p[field]) or not 0 < p[field] < 1:
            raise ValueError(f'{field} must be finite and in (0,1)')
    return p


def check_environment():
    from qc_versions import PACKAGE_VERSIONS, PYTHON_VERSION
    mismatches = []
    if platform.python_version() != PYTHON_VERSION:
        mismatches.append('python=' + platform.python_version() + ', expected ' + PYTHON_VERSION)
    for package, expected in PACKAGE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = 'MISSING'
        if actual != expected:
            mismatches.append(f'{package}={actual}, expected {expected}')
    if mismatches:
        raise ValueError('QC environment mismatch; run setup_qc.cmd --check: ' + '; '.join(mismatches))


def bootstrap():
    from qc_versions import ENVIRONMENT_NAME
    found = subprocess.run([sys.executable, '-m', 'conda', 'env', 'list', '--json'],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
    if found.returncode:
        raise RuntimeError('Conda env list failed: ' + found.stderr.strip())
    matches = [Path(p) for p in json.loads(found.stdout)['envs']
               if Path(p).name == ENVIRONMENT_NAME]
    if len(matches) != 1:
        raise RuntimeError('Expected one ' + ENVIRONMENT_NAME + ' environment; run setup_qc.cmd --check')
    target = matches[0] / ('python.exe' if os.name == 'nt' else 'bin/python')
    if not target.is_file():
        raise FileNotFoundError(target)
    return subprocess.run([str(target), '-B', str(Path(__file__).resolve()), '--from-cmd'],
                          cwd=ROOT).returncode


def new_output(raw):
    from runtime_identity import validate_output_path
    if raw.strip():
        out = validate_output_path(raw.strip())
    else:
        tag = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        out = validate_output_path(ROOT / 'outputs' / ('iqae_sparse_complete_' + tag))
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}; use a new directory')
    return out


def write_new(path, value):
    from runtime_identity import normalize
    with path.open('x', encoding='utf8') as stream:
        json.dump(normalize(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def prepared(out, p):
    from models import load_model
    from runtime_identity import source_manifest, software_versions
    document = {
        'protocol': p, 'config': asdict(load_model(p['model'])),
        'production_sources': source_manifest(), 'versions': software_versions(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_used_for_power_or_stop': False,
        'preselected_before_this_run_counts': True,
        'prior_budget_limited_run_retained': True, 'prior_counts_reused': False,
        'cache_policy': 'Actual complex-state prefix only; fresh binomial batches; no pruning or normalization.',
        'scientific_scope': 'Short actual-gate sparse IQAE; conditional statistical precision, no full error certification.',
    }
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'protocol.json', document)
    print('[PREPARED] ' + str(out), flush=True)
    return document


def load_verified_protocol(out):
    from runtime_identity import source_manifest, software_versions, normalize
    from models import load_model
    document = json.loads((out / 'protocol.json').read_text(encoding='utf8'))
    if document['driver_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Driver changed after preparation')
    if document['production_sources'] != source_manifest():
        raise ValueError('Source identity changed after preparation')
    if document['versions'] != software_versions():
        raise ValueError('Software identity changed after preparation')
    p = document['protocol']
    if p != protocol_from_env(os.environ):
        raise ValueError('Protocol differs from the submitted parameter snapshot')
    if document['config'] != normalize(asdict(load_model(p['model']))):
        raise ValueError('Model changed after preparation')
    return document


class BudgetExceeded(RuntimeError):
    status = 'budget_limited'


def checked_inverse_error(state):
    error = abs(state.get(0, 0) - 1)**2 + math.fsum(abs(a)**2 for key, a in state.items() if key)
    if not math.isfinite(error) or error > 2e-11:
        raise ValueError('A inverse A check failed')
    return error


class ActualGates:
    def __init__(self, Aops, Qops, objective, work_bits, *, max_states, max_gates):
        self.Aops = Aops; self.Qops = Qops; self.objective = objective
        self.work_mask = sum(1 << b for b in work_bits)
        self.max_states = max_states; self.max_gates = max_gates
        self.used = 0; self.peak = 1; self.cache = {}; self.evolutions = []
        self.highest = -1; self.prefix = None; self.active = None

    def evolve(self, ops, state):
        from reference_engine import apply_instruction
        if self.used + len(ops) > self.max_gates:
            raise BudgetExceeded('actual primitive gate budget')
        for op in ops:
            state = apply_instruction(state, op)
            self.used += 1; self.peak = max(self.peak, len(state))
            if len(state) > self.max_states:
                raise BudgetExceeded('complex basis-state budget')
        return state

    def probability(self, k):
        if k in self.cache:
            return self.cache[k]
        start = time.perf_counter(); before = self.used
        reuse = self.highest >= 0 and k > self.highest
        needed = ((k - self.highest) * len(self.Qops) if reuse
                  else len(self.Aops) + k * len(self.Qops))
        self.active = {'requested_power': k, 'prefix_power': self.highest if reuse else None,
                       'required_additional_gates': needed}
        if self.used + needed > self.max_gates:
            raise BudgetExceeded('actual gate budget before requested power')
        if reuse:
            state = self.prefix; first = self.highest
        else:
            state = self.evolve(self.Aops, {0: 1+0j}); first = 0
        for _ in range(first, k):
            state = self.evolve(self.Qops, state)
        norm = math.fsum(abs(a)**2 for a in state.values())
        probability = math.fsum(abs(a)**2 for key, a in state.items() if (key >> self.objective) & 1)
        work = math.fsum(abs(a)**2 for key, a in state.items() if key & self.work_mask)
        if (not all(math.isfinite(v) for v in (probability, norm, work))
                or not 0 <= probability <= 1 or abs(norm - 1) > 2e-11 or work > 2e-11):
            raise ValueError(f'invalid unnormalized actual state: p={probability}, norm={norm}, work={work}')
        if k > self.highest:
            self.highest = k; self.prefix = state
        self.cache[k] = probability
        self.evolutions.append({
            'power': k, 'actual_gate_probability': probability, 'raw_norm': norm,
            'norm_residual': abs(norm - 1), 'work_nonzero_probability': work,
            'seconds': time.perf_counter() - start,
            'primitive_gate_applications': self.used - before, 'peak_basis_so_far': self.peak,
            'final_basis': len(state), 'reused_prefix_power': first if reuse else None,
        })
        self.active = None
        return probability


def estimate_with_protocol(sample, p):
    from iqae import estimate_relative_iqae
    return estimate_relative_iqae(
        sample, relative_error=p['relative_error'], confidence=p['confidence'],
        shots=p['shots_per_round'], max_shots=p['max_shots'],
        max_queries=p['max_queries'], max_power=p['max_power'],
    )


def worker(out):
    import numpy as np
    from models import load_model
    from numerical_runtime import configure_blas_threads
    from unitary_circuit import build_unitary_circuit_qiskit
    from reference_engine import flattened, registers
    from runtime_identity import circuit_fingerprint
    from supervised import _display_record
    from iqae import OracleSpec, build_grover_operator_for_spec
    document = load_verified_protocol(out); p = document['protocol']
    configure_blas_threads(1); begin = time.perf_counter(); cpu = time.process_time()
    cfg = load_model(p['model'])
    A, _ = build_unitary_circuit_qiskit(cfg, p['steps'], kernel_profile=p['profile'])
    regs = registers(A)
    Q, obj, _ = build_grover_operator_for_spec(
        A_circuit=A, status_qubits=regs['status'], g_qubits=regs['g'], spec=OracleSpec(good=p['objective']))
    marked = Q.state_preparation; inverse = A.inverse()
    engine = ActualGates(list(flattened(marked)), list(flattened(Q)), obj[0], regs['work'],
                         max_states=p['max_states'], max_gates=p['max_actual_primitive_gate_applications'])
    batches = []
    base = {
        'protocol': document,
        'fingerprints': {'A': circuit_fingerprint(A), 'A_inverse': circuit_fingerprint(inverse),
                         'marked_A': circuit_fingerprint(marked), 'Q': circuit_fingerprint(Q)},
        'qubits': marked.num_qubits,
        'primitive_components': {'marked_A': len(engine.Aops), 'Q': len(engine.Qops)},
        'primitive_unit': 'Operations from reference_engine.flattened; not hardware two-qubit gates.',
        'full_pipeline_certified': False,
    }
    write_new(out / 'circuit_identity.json', base)

    def progress(status, **extra):
        _display_record(out / 'progress.json', {
            'status': status, 'measurement_batches': batches, 'actual_evolutions': engine.evolutions,
            'active_power': engine.active, 'actual_primitive_gate_applications': engine.used,
            'peak_basis': engine.peak, **extra,
        })

    try:
        state = engine.evolve(list(flattened(A)), {0: 1+0j})
        state = engine.evolve(list(flattened(inverse)), state)
        inverse_error = checked_inverse_error(state); del state
        rng = np.random.default_rng(p['seed'])

        def sample(k, n):
            progress('evolving', requested_power=k)
            print(f'[EVOLVE] power={k} shots={n}', flush=True)
            try:
                probability = engine.probability(k)
            except BaseException as exc:
                progress('evolution_failed', error=repr(exc)); raise
            hits = int(rng.binomial(n, probability))
            batches.append({'power': k, 'shots': n, 'success_count': hits,
                            'actual_gate_probability': probability})
            progress('batch_returned')
            print(f'[ROUND] {len(batches)} power={k} shots={n} hits={hits} actual_gates={engine.used}', flush=True)
            return hits

        result = estimate_with_protocol(sample, p)
        # Independent reference and amplification formula are posthoc only.
        from reference_solver import reference
        ref = reference(cfg, p['steps']); reference_p = ref['rows'][-1]['probabilities'][1]
        theta = math.asin(math.sqrt(reference_p))
        for row in engine.evolutions:
            row['posthoc_amplification_error'] = abs(
                row['actual_gate_probability'] - math.sin((2 * row['power'] + 1) * theta)**2)
        interval = result['confidence_interval']; estimate = result['estimate']
        last = result['last_valid_interval']
        bound = None if last[0] <= 0 else (last[1] - last[0]) / (2 * last[0])
        final = {
            **base, 'status': result['status'], 'estimator': result,
            'measurement_batches': batches, 'actual_evolutions': engine.evolutions,
            'actual_primitive_gate_applications': engine.used, 'peak_basis': engine.peak,
            'unfinished_power': engine.active, 'A_inverse_state_error_squared': inverse_error,
            'last_relative_midpoint_bound': bound, 'posthoc_reference': ref,
            'posthoc_reference_p_D': reference_p,
            'posthoc_relative_error': None if estimate is None else abs(estimate - reference_p) / reference_p,
            'posthoc_interval_contains_reference': None if interval is None else interval[0] <= reference_p <= interval[1],
            'seconds': time.perf_counter() - begin, 'process_cpu_seconds': time.process_time() - cpu,
            'numerical_scope': 'Unpruned complex128 actual gates; empirical residual checks, no rigorous accumulated roundoff enclosure.',
            'reused_prefix_cost_not_discounted_from_estimator_queries': True,
        }
        write_new(out / 'result.json', final); progress(result['status'])
        print(json.dumps({'status': result['status'], 'reason': result['reason'],
                          'estimate': estimate, 'CI': interval, 'relative_bound': bound,
                          'shots': result['costs']['shots'],
                          'queries_A_and_inverse': result['costs']['queries_A_and_inverse']},
                         ensure_ascii=True), flush=True)
        return 0 if result['status'] == 'completed' else 2
    except BaseException as exc:
        progress('failed', error=repr(exc))
        write_new(out / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                       'actual_primitive_gate_applications': engine.used,
                                       'peak_basis': engine.peak, 'batches': batches})
        raise


def launch(out, p):
    from supervised import run_limited
    result = run_limited(
        [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--out', str(out)],
        cwd=ROOT, output_dir=out / 'supervision', memory_bytes=p['memory_mib'] * 1024**2,
        threads=1, blas_threads=1, heartbeat_sec=60, resource_sample_sec=1,
        resource_publish_sec=60, echo_output=True,
    )
    print(json.dumps({'supervision_status': result['status'],
                      'worker_returncode': result['returncode'],
                      'reaped': result['all_processes_reaped']}, ensure_ascii=True), flush=True)
    return 0 if result['status'] == 'passed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--bootstrap-from-cmd', action='store_true')
    modes.add_argument('--from-cmd', action='store_true')
    modes.add_argument('--worker', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.bootstrap_from_cmd:
        return bootstrap()
    if args.worker:
        if os.environ.get('VRQO_SUPERVISED') != '1' or args.out is None:
            raise ValueError('Worker must be launched through the memory/process supervisor')
        check_environment()
        return worker(args.out.resolve())
    if args.out is not None:
        raise ValueError('Set OUTDIR in run.cmd; --out is internal only')
    p = protocol_from_env(os.environ)
    dry = os.environ.get('DRY_RUN', '1').strip()
    if dry not in ('0', '1'):
        raise ValueError('DRY_RUN must be 0 or 1')
    check_environment()
    out = new_output(os.environ.get('OUTDIR', ''))
    from models import load_model
    load_model(p['model'])
    print('[CONFIG] complete native-gate sparse IQAE; one worker; no MPS or shards', flush=True)
    print(json.dumps({'protocol': p, 'output': str(out), 'dry_run': dry == '1'},
                     ensure_ascii=True, indent=2), flush=True)
    if dry == '1':
        print('[DRY RUN OK] No output directory, evolution or random counts created.', flush=True)
        return 0
    prepared(out, p)
    return launch(out, p)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print('[ERROR] ' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
