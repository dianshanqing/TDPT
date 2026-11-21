"""Command-line entry point for MC, circuit, TEST, IQAE, and diagnostics."""


from __future__ import annotations

import argparse
from dataclasses import replace
import os

# Set the worker's library teams before NumPy loads; this is process-local.
for _thread_var in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_var] = (os.environ.get('VRQO_BLAS_THREADS','1')
        if _thread_var in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS') else '1')
import sys
import csv
import json
import hashlib
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

from config import default_config
from mc import run_mc
from plotting import save_mc_plots
from stats_utils import binom_se
from runtime_identity import (validate_output_path, assert_new_file, prepare_run_directory,
    write_json_new, new_run_id, make_identity, circuit_fingerprint, apply_backend_options,
    validate_cache_metadata, normalize, digest, source_manifest, software_versions,validate_completed_result)
import supervised

PROJECT_ROOT = Path(__file__).resolve().parent
REVISION_PHASES = ('screen', 'reference', 'validate', 'mps_probe')


def build_unitary_circuit_qiskit(*args, **kwargs):
    from unitary_circuit import build_unitary_circuit_qiskit as build
    return build(*args, **kwargs)


def estimate_resources(*args, **kwargs):
    from unitary_circuit import estimate_resources as estimate
    return estimate(*args, **kwargs)


def resolve_kernel_profile(*args, **kwargs):
    from unitary_circuit import resolve_kernel_profile as resolve
    return resolve(*args, **kwargs)


def _project_input(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT/path).resolve()


def _load_config(args):
    if getattr(args, 'config', None):
        from models import load_model
        return load_model(args.config)
    return default_config()


def _configure_cpu_budget(args):
    if args.cpu_budget < 1:
        raise ValueError('CPU budget must be positive')
    try:
        import psutil
        available = len(psutil.Process().cpu_affinity())
    except (ImportError, AttributeError, OSError):
        available = os.cpu_count() or 1
    effective = args.cpu_budget if getattr(args, 'dry_run', False) else min(args.cpu_budget, available)
    blas=getattr(args,'blas_threads',1)
    if type(blas) is not int or blas<1:raise ValueError('BLAS threads must be a positive integer')
    nested=args.aer_threads*(args.mps_threads or 1)*blas
    if args.aer_threads < 1 or args.aer_threads > effective or (
            args.mps_threads is not None and not 1 <= args.mps_threads <= effective) or (
            nested > effective):
        raise ValueError('Aer * MPS * BLAS threads must fit the effective CPU budget')
    # Libraries have independent teams. Keep OpenMP separate from BLAS; the
    # actual Aer MPS pragmas use their explicit mps_omp_threads setting.
    os.environ['OMP_NUM_THREADS']=str(args.aer_threads)
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]=str(blas)
    for key in ('NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
        os.environ[key] = '1'
    from numerical_runtime import configure_blas_threads, numerical_libraries
    libraries_before=numerical_libraries()
    applied=[] if getattr(args,'dry_run',False) else configure_blas_threads(blas)
    args.resource_settings = {'requested_cpu_budget': args.cpu_budget,
        'effective_cpu_budget': effective, 'workers': 1, 'blas_threads': blas,
        'nested_thread_upper_bound':nested,'loaded_libraries_before':libraries_before,
        'blas_application':applied,'loaded_libraries_after':numerical_libraries(),
        'available_logical_cpus': available, 'host_fits_request': args.cpu_budget <= available,
        'allocation_performed': not getattr(args, 'dry_run', False),
        'aer_threads': args.aer_threads, 'mps_threads': args.mps_threads or 1,
        'deadline_environment': os.environ.get('VRQO_DEADLINE_MONOTONIC')}
    return args.resource_settings


def phase_revision(args):
    if args.phase == 'reference':
        from reference_solver import reference
        cfg = _apply_xt_overrides(_load_config(args), args)
        result = reference(cfg, args.steps)
        result.update(status='complete', steps=args.steps, phase='reference',
            run_id=args.run_id, resource_settings=args.resource_settings,
            identity=make_identity(config=cfg, profile=None, compile_options=None,
                measurement={'order':['survive','detect','absorb','boundary']},
                backend={'method':'independent_float64_recurrence'}, phase='reference',
                extra={'steps':args.steps}))
        write_json_new(Path(args.outdir)/'result.json', result)
    elif args.phase == 'screen':
        from screening import run_screen
        result = run_screen(_project_input(args.suite), args.outdir, args.cpu_budget)
    else:
        from validation import run_validation, run_mps_probe
        budget=args.resource_settings['effective_cpu_budget']
        if args.phase == 'validate':
            result = run_validation(_project_input(args.suite), args.outdir, budget)
        else:
            if args.perf_prefix_gates:
                from mps_performance import run_prefix_probe
                result=run_prefix_probe(_project_input(args.suite),args.outdir,budget,
                    aer_threads=args.aer_threads,mps_threads=args.mps_threads,
                    blas_threads=args.blas_threads,prefix_gates=args.perf_prefix_gates,
                    reference_path=args.perf_reference)
            else:
                result = run_mps_probe(_project_input(args.suite), args.outdir, budget,
                    aer_threads=args.aer_threads, mps_threads=args.mps_threads,
                    blas_threads=args.blas_threads)
    terminal={'phase':args.phase,'outdir':args.outdir,
              'status':result.get('status','complete'),
              **{k:result.get(k,'not_assessed') for k in ('execution_status','rare_accuracy_status','pilot_accuracy_status','strict_status')}}
    if result.get('science_status')=='not_assessed_prefix':
        terminal.update(science_status=result['science_status'],
                        normalization_status=result.get('normalization_status','not_evaluated'),
                        comparison_status=result.get('comparison',{}).get('status','not_evaluated'))
    print(json.dumps(terminal, ensure_ascii=False))
    return 0 if result.get('all_passed', result.get('passed', True)) else 3

class _TeeStdout:

    def __init__(self, fp, original):
        self.fp = fp
        self.original = original

    def write(self, s):
        self.fp.write(s); self.fp.flush()
        self.original.write(s); self.original.flush()

    def flush(self):
        self.fp.flush()
        self.original.flush()

@contextmanager
def _tee_to_file(log_path: str):
    log_path = str(assert_new_file(log_path))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    orig = sys.stdout
    fp = open(log_path, 'x', encoding='utf-8')
    try:
        with fp:
            sys.stdout = _TeeStdout(fp, orig)
            try:
                yield
            finally:
                sys.stdout = orig
    finally:
        from result_storage import finalize_log
        finalize_log(log_path)

def _default_run_outdir(phase: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join('outputs', f'run_{phase}_{ts}_{new_run_id()[:12]}')

def _log_ts(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

@contextmanager
def _timed_block(title: str, timings=None, key=None):
    t0 = time.monotonic()
    _log_ts(f"START: {title}")
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        if timings is not None:
            timings[key or title] = dt
        _log_ts(f"END  : {title} (elapsed {dt:.2f}s)")

def _effective_kernel_profile(args) -> str:

    name = str(getattr(args, "kernel_profile", "") or "").strip()
    return name  # Empty selects the production baseline; explicit historical names retain meaning.

def _resolved_kernel_profile(args):
    return resolve_kernel_profile(
        kernel_profile=_effective_kernel_profile(args),
        use_dir_tape=bool(getattr(args, "dir_tape", False)),
        use_absorb_tape=bool(getattr(args, "absorb_tape", False)),
        use_term_tape=bool(getattr(args, "term_tape", False)),
        use_scatter_tape=bool(getattr(args, "scatter_tape", False)),
        full_history_mode=bool(getattr(args, "full_history_mode", False)),
    )

def _kernel_profile_flags(profile):
    return {
        "kernel_profile": str(profile.name),
        "dir_tape": bool(profile.use_dir_tape),
        "absorb_tape": bool(profile.use_absorb_tape),
        "term_tape": bool(profile.use_term_tape),
        "scatter_tape": bool(profile.use_scatter_tape),
        "full_history_mode": bool(profile.full_history_mode),
    }

def _thread_settings(backend_state=None):
    effective=(backend_state or {}).get('effective') or {}
    settings={'aer_max_parallel_threads':effective.get('max_parallel_threads'),
        'mps_omp_threads':effective.get('mps_omp_threads'),
        'environment':{name:os.environ.get(name) for name in ('OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS')}}
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        settings['loaded_threadpools']=None
    else:
        settings['loaded_threadpools']=normalize(threadpool_info())
    return settings


def _sampler_timeout(args):
    value=getattr(args,'iqae_max_call_sec',None)
    if value is None or value==0:
        return None  # Explicit zero retains the legacy CLI unlimited spelling.
    if not np.isfinite(value) or value<0:
        raise ValueError('iqae_max_call_sec must be finite and positive, or omitted/zero')
    return float(value)

def _derive_chunk_seed(base_seed: Optional[int], *, abs_shot_offset: int, chunk_shots: int) -> Optional[int]:

    if base_seed is None:
        return None

    mask64 = ((1 << 64) - 1)
    x = int(base_seed) & mask64
    x ^= ((int(abs_shot_offset) + 0x9E3779B97F4A7C15) & mask64)
    x ^= (((int(chunk_shots) + 1) * 0xBF58476D1CE4E5B9) & mask64)
    x &= mask64

    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & mask64
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & mask64
    x ^= (x >> 31)

    return int(x & 0x7FFFFFFF)

def _normalize_meas_order(meas_order_obj):

    out = []
    if meas_order_obj is None:
        return out
    for item in meas_order_obj:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), int(item[1])))
        else:
            out.append(tuple(item))
    return out

def _aer_run_counts_chunked(sim, circuit, *, total_shots: int, shot_chunk: int, heartbeat_sec: float, hb_tag: str, seed: Optional[int] = None, shard_id: int = -1, global_shot_offset: int = 0, progress_state: Optional[dict] = None):

    counts_total: Dict[str, int] = {}
    done = 0

    if progress_state is not None:
        progress_state.update({'stage': 'aer_run', 'shots_done': 0, 'shots_total': int(total_shots), 'shots_inflight': 0})

    chunk = int(shot_chunk) if (shot_chunk is not None and int(shot_chunk) > 0) else 0

    while done < total_shots:
        remain = int(total_shots - done)
        abs_shot_offset = int(global_shot_offset) + int(done)
        if chunk <= 0:
            cur = remain
        else:

            mod = int(abs_shot_offset) % int(chunk)
            to_boundary = (int(chunk) - mod) if mod != 0 else int(chunk)
            cur = min(remain, to_boundary)

        run_kwargs = {}
        chunk_seed = _derive_chunk_seed(seed, abs_shot_offset=int(abs_shot_offset), chunk_shots=int(cur))
        if chunk_seed is not None:
            run_kwargs["seed_simulator"] = int(chunk_seed)

        if progress_state is not None:
            progress_state.update({'stage': 'aer_run', 'shots_done': int(done), 'shots_total': int(total_shots), 'shots_inflight': int(cur)})
        supervised.publish_progress('aer_submit',shots_completed=done,shots_total=int(total_shots),
            shots_inflight=int(cur),sample_start=abs_shot_offset,shard_id=shard_id,tag=hb_tag)
        chunk_started=time.monotonic()
        job = sim.run(circuit, shots=int(cur), **run_kwargs)
        res = job.result()
        chunk_elapsed=time.monotonic()-chunk_started
        if not getattr(res, 'success', True):raise RuntimeError('Backend result reports failure')
        counts = res.get_counts()
        if any(type(v) is not int or v<0 for v in counts.values()) or sum(counts.values())!=cur:
            raise ValueError('Backend counts are not complete integer shots')
        if progress_state is not None:
            progress_state.setdefault('chunk_records',[]).append({
                'sample_start':int(abs_shot_offset),'sample_stop':int(abs_shot_offset+cur),
                'shots':int(cur),'derived_seed':chunk_seed,'status':'complete',
                'elapsed_seconds':chunk_elapsed,
                'backend_metadata':[normalize(getattr(item,'metadata',{})) for item in getattr(res,'results',[])]})
        for k, v in counts.items():
            kk = str(k)
            counts_total[kk] = int(counts_total.get(kk, 0)) + int(v)

        done += int(cur)

        if progress_state is not None:
            progress_state.update({'stage': 'aer_run', 'shots_done': int(done), 'shots_total': int(total_shots), 'shots_inflight': 0})
        supervised.publish_progress('aer_chunk_complete',shots_completed=done,shots_total=int(total_shots),
            shots_inflight=0,last_chunk_elapsed_seconds=chunk_elapsed,shard_id=shard_id,tag=hb_tag)
        _log_ts(f"[PROG] {hb_tag} shots={done}/{total_shots}")

    return counts_total

def _write_results(outdir: str, payload: dict):
    if payload.get('phase') in ('test','mc'):validate_completed_result(payload)
    outdir = str(validate_output_path(outdir))
    p_json = os.path.join(outdir, 'results.json')
    p_txt = os.path.join(outdir, 'results.txt')
    assert_new_file(p_json); assert_new_file(p_txt)
    write_json_new(p_json, payload)
    with open(p_txt, 'x', encoding='utf-8') as f:
        for k, v in payload.items():
            f.write(f"{k}: {v}\n")
    return p_json, p_txt


def _attach_run_metadata(args, payload, identity):
    n=int(args.histories if args.phase=='mc' else args.shots)
    start=int(getattr(args,'global_shot_offset',0))
    sid=max(0,int(getattr(args,'shard_id',-1)))
    total=getattr(args,'total_samples',None)
    root=getattr(args,'root_seed',None)
    payload.update(status=payload.get('status','complete'),run_id=getattr(args,'run_id','') or new_run_id(),
        identity=identity,request_hash=getattr(args,'request_hash','') or identity['identity_hash'],
        sampling={'sample_start':start,'sample_stop':start+n,'total_samples':n if total is None else int(total),
                  'num_shards':int(getattr(args,'num_shards',1)),'shard_id':sid,
                  'root_seed':int(args.seed if root is None else root),'actual_seed':int(args.seed),
                  'seed_strategy':'splitmix64(root_seed, abs_shot_offset, chunk_shots)' if args.phase=='test' else 'root_seed + shard_id',
                  'chunk':int(getattr(args,'aer_shot_chunk',0)) if args.phase=='test' else None})
    payload['shard_id']=sid
    return payload


def invocation_identity(args):
    """Inputs used by the runner to reject changed resumes before any submission."""
    ignored={'outdir','run_id','request_hash','root_seed','total_samples','shard_id','num_shards',
             'global_shot_offset','shard_seed_offset','shots','histories','seed','compiled_qpy',
             'save_compiled_qpy','compile_only','internal_worker'}
    cfg=_apply_xt_overrides(_load_config(args),args)
    return {'arguments':{k:v for k,v in vars(args).items() if k not in ignored},'config':normalize(cfg),
            'actual_profile':normalize(_resolved_kernel_profile(args)) if args.phase not in ('mc',*REVISION_PHASES) else None,
            'sources':source_manifest(),'versions':software_versions()}

def _format_energy_hist(cfg, hist: np.ndarray) -> str:
    b = cfg.groups.boundaries
    lines = []
    for g in range(cfg.groups.n_groups):
        lo = b[g + 1]
        hi = b[g]
        lines.append(f"  g={g}  [{lo:.2f},{hi:.2f}] : {hist[g]:.4f}")
    return "\n".join(lines)

def _apply_xt_overrides(cfg, args):

    if getattr(args, "xt_no_interaction", False):

        mats = cfg.mats

        def _zero_p_int(mat):
            return replace(mat, p_int=[0.0 for _ in mat.p_int])

        mats2 = replace(
            mats,
            concrete=_zero_p_int(mats.concrete),
            tungsten=_zero_p_int(mats.tungsten),
            air=_zero_p_int(mats.air),
        )
        cfg = replace(cfg, mats=mats2)

    return cfg

def phase_mc(args) -> int:
    cfg = _load_config(args)
    cfg = _apply_xt_overrides(cfg, args)
    supervised.publish_progress('mc_sampling',scope='mc_histories',
        histories_completed=0,histories_total=args.histories)
    res = run_mc(
        cfg,
        n_steps=args.steps,
        histories=args.histories,
        seed=args.seed,
        unitary_matched_energy=bool(args.unitary_matched_energy),
    )

    print("=" * 72)
    print("Phase 1: Classical Monte Carlo baseline (Paper-1)")
    print(f"Model: {getattr(cfg, 'model', 'legacy_duct')}, grid={cfg.geom.nx}x{cfg.geom.ny}x{cfg.geom.nz}, detector x={cfg.geom.detector_x}")
    print("=" * 72)
    print(f"steps      : {args.steps}")
    print(f"histories  : {args.histories:,}")
    print(f"seed       : {args.seed}")
    print(f"unitary_matched_energy : {bool(args.unitary_matched_energy)}")
    print("-")
    print(f"P(detect)  : {res.p_detect:.6f}  (SE={res.se_detect:.3e})")
    print(f"P(absorb)  : {res.p_absorb:.6f}  (SE={res.se_absorb:.3e})")
    print(f"P(boundary): {res.p_boundary:.6f}  (SE={res.se_boundary:.3e})")
    print(f"P(survive) : {res.p_survive:.6f}  (SE={res.se_survive:.3e})")
    print("-")
    if res.counts["detect"] > 0:
        print("Energy composition at detector (normalized):")
        print(_format_energy_hist(cfg, res.energy_hist_detect))
    else:
        print("No detector hits; energy histogram is empty.")

    if args.plot:
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        paths = save_mc_plots(cfg, res, outdir=outdir)
        print("-")
        print("Saved figures:")
        for p in paths:
            print(" ", p)

    payload = {
        'phase': 'mc',
        'steps': int(args.steps),
        'histories': int(args.histories),
        'seed': int(args.seed),
        'unitary_matched_energy': bool(args.unitary_matched_energy),
        'p_detect': float(res.p_detect),
        'p_absorb': float(res.p_absorb),
        'p_boundary': float(res.p_boundary),
        'p_survive': float(res.p_survive),
        'se_detect': float(res.se_detect),
        'se_absorb': float(res.se_absorb),
        'se_boundary': float(res.se_boundary),
        'se_survive': float(res.se_survive),
        'energy_hist_detect': res.energy_hist_detect.tolist(),
        'counts': res.counts,
    }
    _attach_run_metadata(args,payload,make_identity(config=cfg,profile=None,compile_options=None,
        measurement={'terminal_counts':list(res.counts)},backend={'method':'numpy_mc'},phase='mc',
        extra={'steps':int(args.steps),'unitary_matched_energy':bool(args.unitary_matched_energy)}))
    _write_results(args.outdir, payload)
    supervised.publish_progress('complete',scope='mc_histories',
        histories_completed=args.histories,histories_total=args.histories)
    return 0

def phase_circuit(args) -> int:
    started = time.perf_counter()
    cfg = _load_config(args)
    cfg = _apply_xt_overrides(cfg, args)
    prof = _resolved_kernel_profile(args)
    est = estimate_resources(
        cfg,
        n_steps=args.steps,
        kernel_profile=prof,
    )

    print("=" * 72)
    print("Phase 2: Build unitary A circuit (Paper-1)")
    print("Key: direction uses 6D coin unitary embedding; per-step interaction uses tape cI[t].")
    print("=" * 72)
    for k in ["xbits", "ybits", "zbits", "gbits", "dir", "pos_qubits", "status", "term", "coin", "cI_tape", "cA_tape_qubits", "dir_tape_qubits", "term_tape_qubits", "tS_tape_qubits", "work", "total_qubits"]:
        if k in est:
            print(f"{k:16s}: {est[k]}")
    if "note" in est:
        print("-")
        print(est["note"])

    try:
        supervised.publish_progress('build',steps=args.steps,phase='circuit')
        qc, _info = build_unitary_circuit_qiskit(
            cfg,
            n_steps=args.steps,
            kernel_profile=prof,
            disable_detector=args.xt_disable_detector,
            disable_boundary=args.xt_disable_boundary,
        )
    except ImportError as e:
        print("\n[ERROR] Failed to build circuit: Qiskit is required for phase=circuit.")
        print("Install: pip install qiskit qiskit-aer qiskit-algorithms")
        print("Exception:", repr(e))
        return 2
    except Exception as e:
        print("\n[ERROR] Failed to build circuit.")
        print("Exception:", repr(e))
        return 2

    print("\nCircuit built.")
    print(f"num_qubits : {qc.num_qubits}")
    try:
        print(f"depth      : {qc.depth()}")
        ops = qc.count_ops()
        top = sorted(ops.items(), key=lambda x: (-x[1], x[0]))[:12]
        print("top ops    :", ", ".join([f"{k}={v}" for k, v in top]))
    except Exception:
        pass

    build_seconds = time.perf_counter() - started
    resources = {'raw_gates':len(qc.data), 'raw_depth':qc.depth(),
        'raw_two_qubit_gates':sum(len(inst.qubits)==2 for inst in qc.data),
        'build_seconds':build_seconds, 'compiled_gates':None,
        'compiled_two_qubit_gates':None, 'compiled_depth':None,
        'compile_seconds':None}
    # Persist the successful build before compilation/fingerprinting can fail.
    write_json_new(Path(args.outdir)/'circuit_build.json', {
        'status':'built_only', 'steps':args.steps, 'config':normalize(cfg),
        'profile':normalize(prof), 'num_qubits':qc.num_qubits,
        'register_map':{reg.name:[qc.find_bit(q).index for q in reg] for reg in qc.qregs},
        'resources':resources, 'sources':source_manifest(), 'versions':software_versions(),
        'gate_fingerprint':None, 'compiled':False})
    compile_options = None
    compiled_fingerprint = None
    if args.aer_safe_basis or getattr(args,'probe_compile',False):
        from qiskit import transpile
        compile_options = {'basis_gates':['rz','sx','x','cx'],
            'optimization_level':0, 'seed_transpiler':args.transpiler_seed, 'num_processes':1}
        if getattr(args,'probe_compile',False):
            compile_options.update(basis_gates=['u','cx'],seed_transpiler=917)
        supervised.publish_progress('compile',phase='circuit',compile_options=compile_options)
        tick = time.perf_counter()
        compiled = transpile(qc, **compile_options)
        resources.update(compiled_gates=len(compiled.data), compiled_depth=compiled.depth(),
            compiled_two_qubit_gates=sum(len(inst.qubits)==2 for inst in compiled.data),
            compile_seconds=time.perf_counter()-tick)
        write_json_new(Path(args.outdir)/'compiled_resources.json', {
            'status':'compiled_identity_pending', 'steps':args.steps,
            'resources':resources, 'compile_options':compile_options,
            'compiled_gate_fingerprint':None, 'simulation_executed':False})
        compiled_fingerprint = circuit_fingerprint(compiled)
    payload = {"phase": "circuit", "steps": int(args.steps), "est": est,
        "num_qubits": int(qc.num_qubits), 'resources':resources,
        'register_map':{reg.name:[qc.find_bit(q).index for q in reg] for reg in qc.qregs},
        'compiled_fingerprint':compiled_fingerprint, 'resource_settings':args.resource_settings}
    payload.update(status='complete',run_id=getattr(args,'run_id','') or new_run_id(),
        identity=make_identity(config=cfg,profile=prof,compile_options=compile_options,measurement=payload['register_map'],
        backend=None,phase='circuit',extra={'steps':int(args.steps),'circuit_fingerprint':circuit_fingerprint(qc)}))
    _write_results(args.outdir, payload)
    return 0

def phase_test(args) -> int:

    cfg = _load_config(args)
    cfg = _apply_xt_overrides(cfg, args)
    prof = _resolved_kernel_profile(args)
    prof_flags = _kernel_profile_flags(prof)

    try:
        sid = int(getattr(args, 'shard_id', -1))
        ns = int(getattr(args, 'num_shards', 1) or 1)
        g = cfg.geom
        s = cfg.source
        print(f"[CFG] test steps={int(args.steps)} shard={sid}/{ns} geom={g.nx}x{g.ny}x{g.nz} src=({s.x0},{s.y0},{s.z0}) det_x={g.detector_x} roi_y=[{g.roi_y0}..{g.roi_y1}] roi_z=[{g.roi_z0}..{g.roi_z1}] kernel_profile={prof.name} dir_tape={prof.use_dir_tape} absorb_tape={prof.use_absorb_tape} term_tape={prof.use_term_tape} scatter_tape={prof.use_scatter_tape} full_history={prof.full_history_mode}")
    except Exception:
        pass

    hb_state = {'stage': 'init'}
    stage_timings = {}
    sid0 = int(getattr(args,'shard_id',-1))
    hb_tag = f"test shard={sid0:03d}"

    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile, QuantumCircuit, ClassicalRegister
    except Exception as e:
        print("[ERROR] Aer not available:", repr(e))
        return 2

    try:
        hb_state['stage'] = 'build'
        supervised.publish_progress('build',shots_completed=0,shots_total=args.shots,steps=args.steps)
        with _timed_block(f"build unitary circuit A (steps={args.steps})",stage_timings,'build'):
            qc, _info = build_unitary_circuit_qiskit(
                cfg,
                n_steps=args.steps,
                kernel_profile=prof,
                disable_detector=args.xt_disable_detector,
                disable_boundary=args.xt_disable_boundary,
            )
    except Exception as e:
        print("[ERROR] Failed to build A circuit.")
        print("Exception:", repr(e))
        return 2

    qcm = QuantumCircuit(*qc.qregs, name=qc.name + "_meas")
    qcm.compose(qc, inplace=True)

    def _find_qreg(name: str):
        for qr in qcm.qregs:
            if qr.name == name:
                return qr
        raise RuntimeError(f"qreg '{name}' not found")
    rg = _find_qreg("g")
    rstatus = _find_qreg("status")

    rx = _find_qreg("x") if getattr(args, 'test_measure_x', False) else None
    xbits = len(rx) if rx is not None else 0

    rwork = _find_qreg("work") if getattr(args, 'debug_measure_work', False) else None
    wbits = len(rwork) if rwork is not None else 0

    rdir = _find_qreg("dir") if getattr(args, 'test_measure_dir', False) else None
    dirbits = len(rdir) if rdir is not None else 0

    rci = _find_qreg("cI") if getattr(args, 'test_measure_coins', False) else None
    cibits = len(rci) if rci is not None else 0

    rca = None
    cabits = 0
    if getattr(args, 'test_measure_coins', False):
        try:
            rca = _find_qreg("cA")
            cabits = len(rca)
        except Exception:
            rca = None
            cabits = 0

    gbits = len(rg)
    sbits = len(rstatus)

    meas_order = [
        ("g", gbits),
        ("status", sbits),
        ("x", xbits),
        ("work", wbits),
        ("dir", dirbits),
        ("cI", cibits),
        ("cA", cabits),
    ]
    total_cbits = sum(n for _, n in meas_order)

    off_g = 0
    off_status = off_g + gbits
    off_x = off_status + sbits
    off_work = off_x + xbits
    off_dir = off_work + wbits
    off_ci = off_dir + dirbits
    off_ca = off_ci + cibits

    qcm.add_register(ClassicalRegister(total_cbits, "c"))
    c = qcm.cregs[0]
    idx = 0
    for q in rg:
        qcm.measure(q, c[idx]); idx += 1
    for q in rstatus:
        qcm.measure(q, c[idx]); idx += 1
    if rx is not None:
        for q in rx:
            qcm.measure(q, c[idx]); idx += 1
    if rwork is not None:
        for q in rwork:
            qcm.measure(q, c[idx]); idx += 1
    if rdir is not None:
        for q in rdir:
            qcm.measure(q, c[idx]); idx += 1
    if rci is not None:
        for q in rci:
            qcm.measure(q, c[idx]); idx += 1
    if rca is not None:
        for q in rca:
            qcm.measure(q, c[idx]); idx += 1

    method = 'matrix_product_state' if (qcm.num_qubits > 32 or args.mps_max_bond is not None or args.mps_trunc is not None) else 'automatic'
    sim = AerSimulator(method=method)
    backend_state = apply_backend_options(sim, _iqae_backend_options(args))
    n_qubits = qcm.num_qubits
    compile_options = {'optimization_level': 0, 'num_processes': 1, 'seed_transpiler': args.transpiler_seed,
        'basis_gates': ['rz','sx','x','cx'] if args.aer_safe_basis else None,
        'skip_transpile': bool(args.skip_transpile)}
    measurement = {'reg_layout': {q.name:len(q) for q in qcm.qregs}, 'order': meas_order,
                   'uncompiled_circuit_fingerprint': circuit_fingerprint(qcm)}
    cache_identity = make_identity(config=cfg,profile=prof,compile_options=compile_options,
        measurement=measurement,backend=backend_state,phase='test',extra={'steps':int(args.steps),
        'disable_detector':bool(args.xt_disable_detector),'disable_boundary':bool(args.xt_disable_boundary)})
    tqc = None
    loaded_compiled = False
    if args.compiled_qpy:
        supervised.publish_progress('load_compiled',shots_completed=0,shots_total=args.shots)
        load_started=time.monotonic()
        import qiskit.qpy as qpy
        cache_path=validate_output_path(args.compiled_qpy)
        with cache_path.open('rb') as f:circuits=qpy.load(f)
        if len(circuits)!=1:raise ValueError('Compiled cache must contain exactly one circuit')
        with Path(str(cache_path)+'.meta.json').open(encoding='utf-8') as f:meta=json.load(f)
        tqc=circuits[0]
        validate_cache_metadata(meta,cache_identity,tqc)
        loaded_compiled=True
        stage_timings['load_compiled']=time.monotonic()-load_started
        print(f'[INFO] Verified complete cache identity and gate fingerprint: {cache_path}')
    print('Backend settings:', json.dumps(backend_state,sort_keys=True))

    if (not loaded_compiled) and (not getattr(args, "skip_transpile", False)):
        hb_state['stage'] = 'transpile'
        supervised.publish_progress('transpile',shots_completed=0,shots_total=args.shots)
        try:
            with _timed_block('transpile',stage_timings,'transpile'):
                if getattr(args,'aer_safe_basis',False):
                    tqc=transpile(qcm,optimization_level=0,basis_gates=['rz','sx','x','cx'],
                        num_processes=1,seed_transpiler=args.transpiler_seed)
                else:
                    tqc=transpile(qcm,optimization_level=0,num_processes=1,seed_transpiler=args.transpiler_seed)
                if 'mcx_vchain' in tqc.count_ops():tqc=tqc.decompose(reps=8)
            print('Compiled top ops:',dict(list(tqc.count_ops().items())[:10]))
        except Exception as error:
            print('[ERROR] Transpile failed:',repr(error))
            traceback.print_exc()
            return 2
    else:
        if tqc is None:
            tqc = qcm
        if loaded_compiled:
            pass
        else:
            print("[WARN] Running without transpile (--skip-transpile). Use only if the circuit is already in safe basis.")

    compiled_fingerprint=circuit_fingerprint(tqc)
    runtime_id=make_identity(config=cfg,profile=prof,compile_options=compile_options,
        measurement=measurement,backend=backend_state,phase='test',extra={'steps':int(args.steps),
        'disable_detector':bool(args.xt_disable_detector),'disable_boundary':bool(args.xt_disable_boundary),
        'circuit_fingerprint':compiled_fingerprint})
    if args.save_compiled_qpy:
        import qiskit.qpy as qpy
        destination=assert_new_file(args.save_compiled_qpy)
        meta_path=assert_new_file(str(destination)+'.meta.json')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as f:qpy.dump([tqc],f)
        write_json_new(meta_path,{'status':'complete','identity':cache_identity,'circuit_fingerprint':compiled_fingerprint})

    if getattr(args, "compile_only", False):
        supervised.publish_progress('compile_complete',shots_completed=0,shots_total=args.shots,
            stage_timings_seconds=stage_timings)
        print("[INFO] compile-only requested; exiting before Aer run.")
        return 0

    try:
        hb_state['stage'] = 'aer_run'
        with _timed_block(f"Aer run (shots={args.shots})",stage_timings,'sampling'):
            counts = _aer_run_counts_chunked(
                sim,
                tqc,
                total_shots=int(args.shots),
                shot_chunk=int(getattr(args, "aer_shot_chunk", 100) or 0),
                heartbeat_sec=float(getattr(args, "aer_heartbeat_sec", 300.0) or 0.0),
                hb_tag=hb_tag,
                seed=int(getattr(args, "seed", 0) or 0),
                shard_id=int(getattr(args, "shard_id", -1)),
                global_shot_offset=int(getattr(args, "global_shot_offset", 0) or 0),
                progress_state=hb_state,
            )
    except Exception as e:
        print('[ERROR] Simulator run failed.')
        print('Exception:', repr(e))
        traceback.print_exc()
        if isinstance(e, MemoryError) or ('bad allocation' in repr(e).lower()) or ('memoryerror' in repr(e).lower()):
            print('[HINT] This often happens when too many shards run in parallel (each Aer MPS instance needs a lot of RAM).')
            print('       Reduce MAX_PARALLEL (e.g. 4~6), or reduce mps_max_bond, or increase system RAM/pagefile.')
        return 2

    dir_count_total: Dict[int, int] = {}
    invalid_dir_total = 0

    ci1 = [0] * int(args.steps) if cibits else []
    ca1 = [0] * int(args.steps) if cabits else []
    ca1_ci0_total = 0
    nonabsorb_with_ca_total = 0
    absorb_without_ca_total = 0

    x_count_total: Dict[int, int] = {}
    x_count_alive: Dict[int, int] = {}

    x_count_detect: Dict[int, int] = {}
    x_count_absorb: Dict[int, int] = {}
    x_count_boundary: Dict[int, int] = {}

    work_nonzero_total = 0
    work_nonzero_alive = 0

    n_alive = 0
    n_detect = 0
    n_absorb = 0
    n_boundary = 0
    g_hist = np.zeros(cfg.groups.n_groups, dtype=int)

    for bitstr, ct in counts.items():
        bits = str(bitstr)[::-1]

        g_bits = bits[off_g:off_g + gbits]
        g_val = 0
        for i, b in enumerate(g_bits):
            if b == '1':
                g_val |= (1 << i)

        s_bits = bits[off_status:off_status + sbits]
        s_val = 0
        for i, b in enumerate(s_bits):
            if b == '1':
                s_val |= (1 << i)

        if xbits:

            x_bits = bits[off_x:off_x + xbits]
            x_val = 0
            for j, b in enumerate(x_bits):
                if b == '1':
                    x_val |= (1 << j)
            x_count_total[x_val] = x_count_total.get(x_val, 0) + ct

            if s_val == 0:
                x_count_alive[x_val] = x_count_alive.get(x_val, 0) + ct
            elif s_val == 1:
                x_count_detect[x_val] = x_count_detect.get(x_val, 0) + ct
            elif s_val == 2:
                x_count_absorb[x_val] = x_count_absorb.get(x_val, 0) + ct
            else:
                x_count_boundary[x_val] = x_count_boundary.get(x_val, 0) + ct

        if wbits:
            off_work = off_x + xbits
            w_bits = bits[off_work:off_work + wbits]
            w_val = 0
            for j, b in enumerate(w_bits):
                if b == '1':
                    w_val |= (1 << j)
            if w_val != 0:
                work_nonzero_total += ct
                if s_val == 0:
                    work_nonzero_alive += ct

        if dirbits:
            dir_bits = bits[off_dir:off_dir + dirbits]
            dir_val = 0
            for j, b in enumerate(dir_bits):
                if b == '1':
                    dir_val |= (1 << j)
            dir_count_total[dir_val] = dir_count_total.get(dir_val, 0) + ct
            if dir_val >= 6:
                invalid_dir_total += ct

        ci_bits = None
        if cibits:
            ci_bits = bits[off_ci:off_ci + cibits]
            for t, b in enumerate(ci_bits[:int(args.steps)]):
                if b == '1':
                    ci1[t] += ct

        if cabits:
            ca_bits = bits[off_ca:off_ca + cabits]
            any_ca = False
            for t, b in enumerate(ca_bits[:int(args.steps)]):
                if b == '1':
                    any_ca = True
                    ca1[t] += ct
                    if ci_bits is not None and t < len(ci_bits) and ci_bits[t] == '0':
                        ca1_ci0_total += ct
            if any_ca and s_val != 2:
                nonabsorb_with_ca_total += ct
            if (not any_ca) and s_val == 2:
                absorb_without_ca_total += ct

        if s_val == 0:
            n_alive += ct
        elif s_val == 1:
            n_detect += ct
            if 0 <= g_val < len(g_hist):
                g_hist[g_val] += ct
        elif s_val == 2:
            n_absorb += ct
        else:
            n_boundary += ct

    shots = int(args.shots)
    p_detect = n_detect / shots
    p_absorb = n_absorb / shots
    p_boundary = n_boundary / shots
    p_survive = n_alive / shots

    se_detect = binom_se(p_detect, shots)
    se_absorb = binom_se(p_absorb, shots)
    se_boundary = binom_se(p_boundary, shots)
    se_survive = binom_se(p_survive, shots)

    print("\nTest (Aer sampling) summary:")
    print(f"steps : {args.steps}")
    print(f"shots : {shots}")
    print(f"P(detect)   ~ {p_detect:.6f}  (SE={se_detect:.3e})")
    print(f"P(absorb)   ~ {p_absorb:.6f}  (SE={se_absorb:.3e})")
    print(f"P(boundary) ~ {p_boundary:.6f}  (SE={se_boundary:.3e})")
    print(f"P(survive)  ~ {p_survive:.6f}  (SE={se_survive:.3e})")

    if wbits:
        p_work_nonzero = work_nonzero_total / shots
        print(f"Work ancilla nonzero rate (all shots): {p_work_nonzero:.6f}")
        if n_alive:
            print(f"Work ancilla nonzero rate (alive only): {work_nonzero_alive/n_alive:.6f}")
        else:
            print('Work ancilla nonzero rate (alive only): undefined (no alive samples)')

    diag = {}
    x_prob_total = None
    x_prob_alive = None
    x_prob_detect = None
    x_prob_absorb = None
    x_prob_boundary = None
    if xbits:
        x_prob_total = {int(k): float(v) / float(shots) for k, v in sorted(x_count_total.items())}
        x_prob_alive = {int(k): float(v) / float(max(1, n_alive)) for k, v in sorted(x_count_alive.items())}
        x_prob_detect = {int(k): float(v) / float(max(1, n_detect)) for k, v in sorted(x_count_detect.items())}
        x_prob_absorb = {int(k): float(v) / float(max(1, n_absorb)) for k, v in sorted(x_count_absorb.items())}
        x_prob_boundary = {int(k): float(v) / float(max(1, n_boundary)) for k, v in sorted(x_count_boundary.items())}

        print("x histogram (total):")
        for xk, pv in x_prob_total.items():
            if pv > 0:
                print(f"  x={xk:02d}: {pv:.6f}")

        print("x histogram (alive | normalized):")
        for xk, pv in x_prob_alive.items():
            if pv > 0:
                print(f"  x={xk:02d}: {pv:.6f}")

        try:
            reach_dist = abs(int(cfg.geom.detector_x) - int(cfg.source.x0))
            diag["reach_dist"] = int(reach_dist)
            diag["unreachable_detect"] = (int(args.steps) < reach_dist and int(n_detect) > 0)
            if diag["unreachable_detect"]:
                print(f"[WARN] unreachable-detect: steps={int(args.steps)} < |det_x-x0|={reach_dist} but n_detect={int(n_detect)}")
            alive_x0 = int(x_count_alive.get(0, 0))
            alive_xmax = int(x_count_alive.get(int(cfg.geom.nx) - 1, 0))
            diag["alive_at_x0"] = alive_x0
            diag["alive_at_xmax"] = alive_xmax
            diag["alive_on_x_boundary"] = ((alive_x0 + alive_xmax) > 0)
            if diag["alive_on_x_boundary"]:
                print(f"[WARN] alive-on-x-boundary: alive_at_x0={alive_x0} alive_at_xmax={alive_xmax}")
        except Exception:
            diag = {}
    work_nonzero_rate_total = None
    work_nonzero_rate_alive = None
    if wbits:
        work_nonzero_rate_total = work_nonzero_total / shots
        work_nonzero_rate_alive = work_nonzero_alive / n_alive if n_alive else None

    reg_layout = {qr.name: len(qr) for qr in qcm.qregs}
    meas_offsets = {
        "g": int(off_g),
        "status": int(off_status),
        "x": int(off_x),
        "work": int(off_work),
        "dir": int(off_dir),
        "cI": int(off_ci),
        "cA": int(off_ca),
    }

    coin_stats = None
    if cibits or cabits:
        coin_stats = {
            "cibits": int(cibits),
            "cabits": int(cabits),
            "ci1": [int(v) for v in ci1],
            "ca1": ([int(v) for v in ca1] if cabits else None),
            "p_ci1": [float(v) / float(shots) for v in ci1],
            "p_ca1": ([float(v) / float(shots) for v in ca1] if cabits else None),
            "ca1_ci0_total": int(ca1_ci0_total),
            "nonabsorb_with_ca_total": int(nonabsorb_with_ca_total),
            "absorb_without_ca_total": int(absorb_without_ca_total),
            "nonabsorb_with_ca_rate": float(nonabsorb_with_ca_total) / float(shots) if shots > 0 else 0.0,
            "absorb_without_ca_rate": float(absorb_without_ca_total) / float(shots) if shots > 0 else 0.0,
            "ca1_ci0_rate": float(ca1_ci0_total) / float(shots) if shots > 0 else 0.0,
        }

    dir_stats = None
    if dirbits:
        dir_stats = {
            "dirbits": int(dirbits),
            "dir_count_total": {str(k): int(v) for k, v in sorted(dir_count_total.items())},
            "invalid_dir_total": int(invalid_dir_total),
            "invalid_dir_rate": float(invalid_dir_total) / float(shots) if shots > 0 else 0.0,
        }

    circuit_signature = {
        "steps": int(args.steps),
        "flags": {
            "kernel_profile": str(prof.name),
            "dir_tape": bool(prof.use_dir_tape),
            "absorb_tape": bool(prof.use_absorb_tape),
            "term_tape": bool(prof.use_term_tape),
            "scatter_tape": bool(prof.use_scatter_tape),
            "full_history_mode": bool(prof.full_history_mode),
            "test_measure_x": bool(xbits),
            "test_measure_coins": bool(getattr(args, "test_measure_coins", False)),
            "test_measure_dir": bool(getattr(args, "test_measure_dir", False)),
            "aer_safe_basis": bool(getattr(args, "aer_safe_basis", False)),
        },
        "reg_layout": reg_layout,
        "meas_order": meas_order,
        "meas_offsets": meas_offsets,
        "sim_method": str(method),
        "num_qubits": int(n_qubits),
    }
    try:
        if tqc is not None:
            op_counts = dict(tqc.count_ops())
            circuit_signature.update({
                "compiled_depth": int(tqc.depth()),
                "compiled_size": int(tqc.size()),
                "compiled_op_counts_top": {str(k): int(v) for k, v in sorted(op_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:20]},
            })
            hobj = {
                "num_qubits": int(tqc.num_qubits),
                "depth": int(tqc.depth()),
                "size": int(tqc.size()),
                "op_counts": {str(k): int(v) for k, v in sorted(op_counts.items(), key=lambda kv: str(kv[0]))},
            }
            circuit_signature["compiled_sig"] = hashlib.sha256(json.dumps(hobj, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        pass

    payload = {
        'phase': 'test',
        'steps': int(args.steps),
        'shots': int(shots),
        'seed': int(getattr(args,'seed',0) or 0),
        'shard_id': int(getattr(args,'shard_id',-1)),
        'num_shards': int(getattr(args,'num_shards',1) or 1),
        'aer_shot_chunk': int(getattr(args, 'aer_shot_chunk', 0) or 0),
        'global_shot_offset': int(getattr(args, 'global_shot_offset', 0) or 0),
        'seed_strategy': 'splitmix64(root_seed, abs_shot_offset, chunk_shots)',
        'kernel_profile': str(prof.name),
        'dir_tape': bool(prof.use_dir_tape),
        'absorb_tape': bool(prof.use_absorb_tape),
        'term_tape': bool(prof.use_term_tape),
        'scatter_tape': bool(prof.use_scatter_tape),
        'full_history_mode': bool(prof.full_history_mode),
        'test_measure_coins': bool(getattr(args, 'test_measure_coins', False)),
        'test_measure_dir': bool(getattr(args, 'test_measure_dir', False)),
        'test_measure_x': bool(xbits),
        'cfg': {
            'nx': int(cfg.geom.nx), 'ny': int(cfg.geom.ny), 'nz': int(cfg.geom.nz),
            'source': {'x0': int(cfg.source.x0), 'y0': int(cfg.source.y0), 'z0': int(cfg.source.z0), 'g0': int(cfg.source.g0), 'dir0': int(getattr(cfg.source,'dir0',0))},
            'detector_x': int(cfg.geom.detector_x),
            'roi': {'y0': int(cfg.geom.roi_y0), 'y1': int(cfg.geom.roi_y1), 'z0': int(cfg.geom.roi_z0), 'z1': int(cfg.geom.roi_z1)},
        },
        'diag': diag,
        'n_detect': int(n_detect),
        'n_absorb': int(n_absorb),
        'n_boundary': int(n_boundary),
        'n_survive': int(shots - n_detect - n_absorb - n_boundary),
        'n_alive': int(n_alive),
        'counts': {'detect': int(n_detect), 'absorb': int(n_absorb), 'boundary': int(n_boundary), 'survive': int(shots - n_detect - n_absorb - n_boundary)},
        'p_detect': float(p_detect),
        'p_absorb': float(p_absorb),
        'p_boundary': float(p_boundary),
        'p_survive': float(p_survive),
        'se_detect': float(se_detect),
        'se_absorb': float(se_absorb),
        'se_boundary': float(se_boundary),
        'se_survive': float(se_survive),
        'x_count_total': {str(k): int(v) for k, v in x_count_total.items()},
        'x_count_alive': {str(k): int(v) for k, v in x_count_alive.items()},
        'x_count_detect': {str(k): int(v) for k, v in x_count_detect.items()},
        'x_count_absorb': {str(k): int(v) for k, v in x_count_absorb.items()},
        'x_count_boundary': {str(k): int(v) for k, v in x_count_boundary.items()},
        'x_prob_total': x_prob_total,
        'x_prob_alive': x_prob_alive,
        'x_prob_detect': x_prob_detect,
        'x_prob_absorb': x_prob_absorb,
        'x_prob_boundary': x_prob_boundary,
        'work_nonzero_rate_total': work_nonzero_rate_total,
        'work_nonzero_rate_alive': work_nonzero_rate_alive,
        'reg_layout': reg_layout,
        'meas_order': meas_order,
        'meas_offsets': meas_offsets,
        'coin_stats': coin_stats,
        'dir_stats': dir_stats,
        'circuit_signature': circuit_signature,
    }
    payload['circuit_fingerprint']=compiled_fingerprint
    payload['raw_measurement_counts']={str(k):int(v) for k,v in counts.items()}
    payload['backend_settings']=backend_state
    payload['thread_settings']=_thread_settings(backend_state)
    payload['stage_timings_seconds']=stage_timings
    payload['chunk_records']=hb_state.get('chunk_records',[])
    payload['work_measurement']={'specified_indices':list(range(len(_find_qreg('work')))),
        'measured_indices':list(range(wbits)), 'coverage':'complete' if wbits else 'unmeasured',
        'nonzero_total_count':int(work_nonzero_total) if wbits else None,
        'nonzero_alive_count':int(work_nonzero_alive) if wbits else None}
    _attach_run_metadata(args,payload,runtime_id)
    _write_results(args.outdir, payload)
    supervised.publish_progress('complete',shots_completed=shots,shots_total=shots,
        stage_timings_seconds=stage_timings)
    return 0

def _print_iqae_run_cfg(args, cfg, backend_options: dict, *, good: str, target_g: Optional[int], outdir: str):

    try:
        geom = getattr(cfg, 'geom', None)
        src = getattr(cfg, 'source', None)
        print('[IQAE_CFG] --------------------------------------------------')
        print(f"[IQAE_CFG] phase=iqae outdir={outdir}")
        if geom is not None and src is not None:
            print(f"[IQAE_CFG] geom={int(geom.nx)}x{int(geom.ny)}x{int(geom.nz)} src=({int(src.x0)},{int(src.y0)},{int(src.z0)}) det_x={int(geom.detector_x)}")
        print(
            '[IQAE_CFG] '
            f"steps={int(args.steps)} shots={int(args.shots)} seed={int(args.seed)} good={good} target_g={target_g} "
            f"mode={args.iqae_mode} relative_error={args.relative_error:.6g} confidence={args.confidence:.6g} "
            f"max_shots={args.iqae_max_shots} max_A_plus_inverse_queries={args.iqae_max_queries} "
            f"max_power={args.iqae_max_grover_power}"
        )
        print(
            '[IQAE_CFG] '
            f"dir_tape={bool(getattr(args, 'dir_tape', False))} absorb_tape={bool(getattr(args, 'absorb_tape', False))} "
            f"term_tape={bool(getattr(args, 'term_tape', False))} scatter_tape={bool(getattr(args, 'scatter_tape', False))} "
            f"full_history={bool(getattr(args, 'full_history_mode', False))} safe_basis={bool(getattr(args, 'aer_safe_basis', False))}"
        )
        print(
            '[IQAE_CFG] '
            f"aer_threads={int(getattr(args, 'aer_threads', 1))} shot_chunk={int(getattr(args, 'iqae_shot_chunk', 100) or 0)} "
            f"heartbeat_sec={float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0):.6g}"
        )
        if backend_options:
            print('[IQAE_CFG] backend_options=' + json.dumps(backend_options, ensure_ascii=False, sort_keys=True))
        else:
            print('[IQAE_CFG] backend_options={}')
        print('[IQAE_CFG] --------------------------------------------------')
    except Exception as e:
        print(f"[IQAE_CFG] WARN: failed to print config block: {repr(e)}")

def _iqae_backend_options(args) -> dict:
    options={'max_parallel_threads':int(getattr(args,'aer_threads',1))}
    if options['max_parallel_threads']<=0:raise ValueError('aer_threads must be positive')
    mps_threads=getattr(args,'mps_threads',None)
    if mps_threads is not None:
        if isinstance(mps_threads,bool) or not isinstance(mps_threads,int) or mps_threads<=0:
            raise ValueError('mps_threads must be a positive integer or omitted')
        options['mps_omp_threads']=mps_threads
    if getattr(args,'aer_safe_basis',False):options['fusion_enable']=False
    memory=int(getattr(args,'aer_max_mem_mb',0))
    if memory<0:raise ValueError('aer_max_mem_mb must be nonnegative')
    if memory:options['max_memory_mb']=memory
    bond=getattr(args,'mps_max_bond',None)
    if bond is not None:
        if bond<0:raise ValueError('mps_max_bond must be nonnegative or omitted')
        options['matrix_product_state_max_bond_dimension']=int(bond)
    trunc=getattr(args,'mps_trunc',None)
    if trunc is not None:
        if not np.isfinite(trunc) or trunc<0:raise ValueError('mps_trunc must be finite and nonnegative or omitted')
        options['matrix_product_state_truncation_threshold']=float(trunc)
    return options


def _is_rare_detect_good_name(good: str) -> bool:
    return str(good).strip().lower().replace("-", "_") in ("rare_detect", "rare_detect_response", "rare_response_detect")

def _make_oracle_spec(args, OracleSpec):
    target_g = None
    if getattr(args, "iqae_target_g", -1) is not None and int(args.iqae_target_g) >= 0:
        target_g = int(args.iqae_target_g)

    good = str(getattr(args, "iqae_good", "detect")).strip().lower().replace("-", "_")
    rare_eta = None
    p_target = None
    if getattr(args, "rare_p_target", None) is not None and float(args.rare_p_target) > 0.0:
        p_target = float(args.rare_p_target)
    if _is_rare_detect_good_name(good):
        rare_eta = float(getattr(args, "rare_eta", -1.0))
        if not (0.0 <= rare_eta <= 1.0):
            raise ValueError("rare_detect objective requires --rare-eta in [0,1]")
        if target_g is not None:
            raise ValueError("rare_detect objective is detect-only in this stage; do not combine with --iqae-target-g")
    return good, target_g, OracleSpec(good=good, target_g=target_g, rare_eta=rare_eta, p_target=p_target)

def _parse_mlae_powers(text: str) -> list:
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        v = int(item)
        if v < 0:
            raise ValueError("MLAE powers must be non-negative")
        vals.append(v)
    if not vals:
        raise ValueError("provide at least one MLAE power")
    return sorted(set(vals))

def phase_iqae(args) -> int:

    try:
        from iqae import run_iqae_small, OracleSpec
    except Exception as e:
        print("\n[ERROR] Failed to import iqae module.")
        print("Exception:", repr(e))
        return 2
    cfg = _load_config(args)
    cfg = _apply_xt_overrides(cfg, args)

    try:
        sid = int(getattr(args, 'shard_id', -1))
        ns = int(getattr(args, 'num_shards', 1) or 1)
        g = cfg.geom
        s = cfg.source
        print(f"[CFG] iqae steps={int(args.steps)} shard={sid}/{ns} geom={g.nx}x{g.ny}x{g.nz} src=({s.x0},{s.y0},{s.z0}) det_x={g.detector_x} dir_tape={bool(getattr(args,'dir_tape',False))} good={str(getattr(args,'iqae_good','detect'))}")
    except Exception:
        pass

    hb_state = {'stage': 'init'}
    sid0 = int(getattr(args,'shard_id',-1))
    hb_tag = f"iqae shard={sid0:03d}"
    stage_timings={}

    try:
        hb_state['stage'] = 'build'
        supervised.publish_progress('build',phase='iqae',steps=args.steps)
        with _timed_block(f"build unitary circuit A (steps={args.steps})",stage_timings,'build'):
            A, _info = build_unitary_circuit_qiskit(
                cfg,
                n_steps=args.steps,
                kernel_profile=_resolved_kernel_profile(args),
            )
    except Exception as e:
        print("[ERROR] Failed to build A circuit.")
        print("Exception:", repr(e))
        return 2

    def _find_qreg(qc, name: str):
        for qr in qc.qregs:
            if qr.name == name:
                return qr
        raise RuntimeError(f"qreg '{name}' not found")

    status_qr = _find_qreg(A, "status")
    g_qr = _find_qreg(A, "g")

    try:
        good, target_g, spec = _make_oracle_spec(args, OracleSpec)
    except Exception as e:
        print("[ERROR] Invalid IQAE objective configuration.")
        print("Exception:", repr(e))
        return 2

    backend_options = _iqae_backend_options(args)

    _print_iqae_run_cfg(args, cfg, backend_options, good=good, target_g=target_g, outdir=str(args.outdir))

    if args.iqae_diagnostic:
        from iqae_diagnostics import run_diagnostic
        record=run_diagnostic(A_circuit=A,status_qubits=status_qr,g_qubits=g_qr,
            spec=spec,powers=args.iqae_powers,outdir=Path(args.outdir)/'diagnostic',
            seed=int(args.seed),mode=args.iqae_diagnostic,backend_options=backend_options,
            max_sparse_states=args.iqae_max_sparse_states,
            max_compiled_gates=args.iqae_max_compiled_gates,compare_to=args.iqae_compare_to,
            case_metadata={'config':normalize(cfg),'model_locator':args.config,
                'steps':args.steps,'profile':_resolved_kernel_profile(args),
                'resource_settings':args.resource_settings,'build_info':normalize(_info),
                'identity':make_identity(config=cfg,profile=_resolved_kernel_profile(args),
                    compile_options={'basis_gates':['rz','sx','x','cx'],'optimization_level':0,'seed_transpiler':int(args.seed)},
                    measurement={'objective':good,'target_g':target_g},backend=backend_options,
                    phase='iqae_fixed_power_diagnostic')})
        write_json_new(Path(args.outdir)/'results.json',record)
        print(json.dumps({'phase':'iqae_fixed_power_diagnostic','status':record.get('status'),
            'estimator_executed':False,'outdir':args.outdir},ensure_ascii=False),flush=True)
        return 0 if record.get('status') in ('complete','passed') else 3

    hb_state['stage'] = 'iqae_run'
    supervised.publish_progress('iqae_run',shots_completed=0,shots_total=None)
    estimator_started=time.monotonic()
    est, meta = run_iqae_small(
        A_circuit=A,
        status_qubits=status_qr,
        g_qubits=g_qr,
        spec=spec,
        epsilon_target=float(getattr(args, 'iqae_eps', 0.05)),
        alpha=float(getattr(args, 'iqae_alpha', 0.05)),
        precision_mode=args.iqae_mode,
        relative_error=args.relative_error,
        confidence=args.confidence,
        max_shots=args.iqae_max_shots,
        max_queries=args.iqae_max_queries,
        shots=int(args.shots),
        use_aer_sampler=bool(getattr(args, 'iqae_use_aer', True)),
        seed=int(args.seed),
        verbose_timing=True,
        backend_options=backend_options,
        debug_sampler=bool(getattr(args, 'iqae_debug', False)),
        sampler_heartbeat_sec=float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0),
        sampler_shot_chunk=int(getattr(args, 'iqae_shot_chunk', 100) or 0),
        sampler_print_ops=bool(getattr(args, 'iqae_debug_print_ops', False)),
        sampler_max_call_seconds=_sampler_timeout(args),
        progress_state=hb_state,
        iqae_max_grover_power=(None if int(getattr(args, 'iqae_max_grover_power', -1)) < 0 else int(getattr(args, 'iqae_max_grover_power'))),
        output_dir=Path(args.outdir)/'ae_worker',
        memory_bytes=None if args.run_memory_mib is None else args.run_memory_mib*1024**2,
    )
    stage_timings['estimator']=time.monotonic()-estimator_started

    ci = None
    try:
        ci = meta.get("confidence_interval", None)
    except Exception:
        ci = None
    if ci is not None:
        try:
            print(f"IQAE confidence interval: [{ci[0]:.6f}, {ci[1]:.6f}] (alpha={meta.get('alpha', None)})")
        except Exception:
            pass

    payload = {
        'phase': 'iqae',
        'status':meta.get('status','failed'),
        'steps': int(args.steps),
        'shots': int(args.shots),
        'cfg': {
            'nx': int(cfg.geom.nx), 'ny': int(cfg.geom.ny), 'nz': int(cfg.geom.nz),
            'source': {'x0': int(cfg.source.x0), 'y0': int(cfg.source.y0), 'z0': int(cfg.source.z0), 'g0': int(cfg.source.g0), 'dir0': int(getattr(cfg.source,'dir0',0))},
            'detector_x': int(cfg.geom.detector_x),
        },
        'good': good,
        'target_g': target_g,
        'rare_eta': meta.get('rare_eta', None),
        'theta_eta': meta.get('theta_eta', None),
        'p_target': meta.get('p_target', None),
        'estimate': None if est is None else float(est),
        'oracle_calls': meta.get('num_oracle_queries', None),
        'wall_clock_time_sec': (meta.get('timing_sec', {}) or {}).get('total', None),
        'circuit_depth': (meta.get('state_preparation_stats', {}) or {}).get('depth', None),
        'gate_count': (meta.get('state_preparation_stats', {}) or {}).get('gate_count', None),
        'qubit_count': (meta.get('state_preparation_stats', {}) or {}).get('num_qubits', None),
        'run_cfg': {
            'seed': int(args.seed),
            'dir_tape': bool(getattr(args, 'dir_tape', False)),
            'absorb_tape': bool(getattr(args, 'absorb_tape', False)),
            'term_tape': bool(getattr(args, 'term_tape', False)),
            'scatter_tape': bool(getattr(args, 'scatter_tape', False)),
            'full_history_mode': bool(getattr(args, 'full_history_mode', False)),
            'kernel_profile': _effective_kernel_profile(args),
            'aer_safe_basis': bool(getattr(args, 'aer_safe_basis', False)),
            'aer_threads': int(getattr(args, 'aer_threads', 1)),
            'mps_threads': getattr(args,'mps_threads',None),
            'mps_max_bond': getattr(args, 'mps_max_bond', None),
            'mps_trunc': getattr(args, 'mps_trunc', None),
            'iqae_shot_chunk': int(getattr(args, 'iqae_shot_chunk', 100) or 0),
            'iqae_heartbeat_sec': float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0),
            'iqae_eps': float(getattr(args, 'iqae_eps', 0.05)),
            'iqae_alpha': float(getattr(args, 'iqae_alpha', 0.05)),
            'precision_mode': args.iqae_mode,
            'relative_error': args.relative_error,
            'confidence': args.confidence,
            'iqae_max_shots': args.iqae_max_shots,
            'iqae_max_queries': args.iqae_max_queries,
            'iqae_max_grover_power': int(getattr(args, 'iqae_max_grover_power', -1)),
        },
        'meta': meta
        }
    payload['thread_settings']=_thread_settings(meta.get('backend_settings'))
    payload['stage_timings_seconds']=stage_timings
    payload.update(run_id=args.run_id,identity=make_identity(config=cfg,profile=_resolved_kernel_profile(args),
        compile_options={'safe_basis':args.aer_safe_basis},measurement={'status_qubits':[A.find_bit(q).index for q in status_qr],
        'g_qubits':[A.find_bit(q).index for q in g_qr],'good':good,'target_g':target_g},
        backend=meta.get('backend_settings',{'requested':backend_options,'effective':None}),phase='iqae',
        extra={'steps':args.steps,'A_fingerprint':circuit_fingerprint(A),'oracle':normalize(spec)}))
    _write_results(args.outdir, payload)

    print("\nIQAE estimate:")
    print(payload)
    return 0 if meta.get('status') in ('complete','completed') and est is not None else 3

def phase_mlae(args) -> int:

    try:
        from iqae import run_fixed_depth_ae_sampling, OracleSpec
    except Exception as e:
        print("\n[ERROR] Failed to import iqae module.")
        print("Exception:", repr(e))
        return 2

    cfg = _load_config(args)
    cfg = _apply_xt_overrides(cfg, args)

    try:
        powers = _parse_mlae_powers(getattr(args, "mlae_powers", "0,1"))
    except Exception as e:
        print("[ERROR] Invalid --mlae-powers.")
        print("Exception:", repr(e))
        return 2

    try:
        good, target_g, spec = _make_oracle_spec(args, OracleSpec)
    except Exception as e:
        print("[ERROR] Invalid MLAE objective configuration.")
        print("Exception:", repr(e))
        return 2

    hb_state = {'stage': 'init'}
    sid0 = int(getattr(args,'shard_id',-1))
    hb_tag = f"mlae shard={sid0:03d}"
    stage_timings={}

    try:
        hb_state['stage'] = 'build'
        supervised.publish_progress('build',phase='mlae',steps=args.steps)
        with _timed_block(f"build unitary circuit A for fixed-depth AE (steps={args.steps})",stage_timings,'build'):
            A, _info = build_unitary_circuit_qiskit(
                cfg,
                n_steps=args.steps,
                kernel_profile=_resolved_kernel_profile(args),
            )
    except Exception as e:
        print("[ERROR] Failed to build A circuit.")
        print("Exception:", repr(e))
        return 2

    def _find_qreg(qc, name: str):
        for qr in qc.qregs:
            if qr.name == name:
                return qr
        raise RuntimeError(f"qreg '{name}' not found")

    status_qr = _find_qreg(A, "status")
    g_qr = _find_qreg(A, "g")
    backend_options = _iqae_backend_options(args)

    try:
        hb_state['stage'] = 'mlae_run'
        supervised.publish_progress('mlae_run',shots_completed=0,shots_total=len(powers)*args.shots)
        estimator_started=time.monotonic()
        result = run_fixed_depth_ae_sampling(
            A_circuit=A,
            status_qubits=status_qr,
            g_qubits=g_qr,
            spec=spec,
            powers=powers,
            shots=int(args.shots),
            seed=int(args.seed),
            verbose_timing=True,
            backend_options=backend_options,
            debug_sampler=bool(getattr(args, 'iqae_debug', False)),
            sampler_heartbeat_sec=float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0),
            sampler_shot_chunk=int(getattr(args, 'iqae_shot_chunk', 100) or 0),
            sampler_print_ops=bool(getattr(args, 'iqae_debug_print_ops', False)),
            sampler_max_call_seconds=_sampler_timeout(args),
            progress_state=hb_state,
            iqae_max_grover_power=(None if args.iqae_max_grover_power<0 else args.iqae_max_grover_power),
            output_dir=Path(args.outdir)/'ae_worker',
            memory_bytes=None if args.run_memory_mib is None else args.run_memory_mib*1024**2,
        )
        stage_timings['estimator']=time.monotonic()-estimator_started
    except Exception as e:
        print("[ERROR] Failed to run fixed-depth AE sampling.")
        print("Exception:", repr(e))
        return 2

    rows = result.get("rows", [])
    meta = result.get("meta", {})
    payload = {
        'phase': 'mlae',
        'status':meta.get('status','failed'),
        'steps': int(args.steps),
        'shots': int(args.shots),
        'powers': [int(x) for x in powers],
        'good': good,
        'target_g': target_g,
        'rare_eta': meta.get('rare_eta', None),
        'theta_eta': meta.get('theta_eta', None),
        'p_target': meta.get('p_target', None),
        'cfg': {
            'nx': int(cfg.geom.nx), 'ny': int(cfg.geom.ny), 'nz': int(cfg.geom.nz),
            'source': {'x0': int(cfg.source.x0), 'y0': int(cfg.source.y0), 'z0': int(cfg.source.z0), 'g0': int(cfg.source.g0), 'dir0': int(getattr(cfg.source,'dir0',0))},
            'detector_x': int(cfg.geom.detector_x),
        },
        'run_cfg': {
            'seed': int(args.seed),
            'kernel_profile': _effective_kernel_profile(args),
            'aer_safe_basis': bool(getattr(args, 'aer_safe_basis', False)),
            'aer_threads': int(getattr(args, 'aer_threads', 1)),
            'mps_threads': getattr(args,'mps_threads',None),
            'mps_max_bond': getattr(args, 'mps_max_bond', None),
            'mps_trunc': getattr(args, 'mps_trunc', None),
            'iqae_shot_chunk': int(getattr(args, 'iqae_shot_chunk', 100) or 0),
            'iqae_heartbeat_sec': float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0),
        },
        'rows': rows,
        'meta': meta,
    }
    payload['thread_settings']=_thread_settings(meta.get('backend_settings'))
    payload['stage_timings_seconds']=stage_timings
    payload.update(run_id=args.run_id,identity=make_identity(config=cfg,profile=_resolved_kernel_profile(args),
        compile_options={'safe_basis':args.aer_safe_basis},measurement={'status_qubits':[A.find_bit(q).index for q in status_qr],
        'g_qubits':[A.find_bit(q).index for q in g_qr],'good':good,'target_g':target_g},
        backend=meta.get('backend_settings',{'requested':backend_options,'effective':None}),phase='mlae',
        extra={'steps':args.steps,'A_fingerprint':circuit_fingerprint(A),'oracle':normalize(spec),'powers':powers}))
    _write_results(args.outdir, payload)

    csv_path = os.path.join(args.outdir, "mlae_summary.csv")
    fieldnames = [
        "m", "oracle_calls", "shots", "success_count", "failure_count",
        "success_probability", "success_se", "circuit_depth", "qubit_count",
        "elapsed_time_sec", "build_power_time_sec",
    ]
    assert_new_file(csv_path)
    with open(csv_path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print("\nFixed-depth AE sampling result:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {csv_path}")
    return 0 if meta.get('status') in ('complete','completed') else 3

def _apply_sharding(args):

    try:
        shard_id = int(getattr(args, "shard_id", -1))
    except Exception:
        shard_id = -1
    if shard_id < 0:
        return
    try:
        setattr(args, "num_shards", max(1, int(getattr(args, "num_shards", 1) or 1)))
    except Exception:
        pass
    try:
        off = int(getattr(args, "shard_seed_offset", 0) or 0)
        if off > 0 and hasattr(args, "seed") and getattr(args, "seed", None) is not None:
            setattr(args, "seed", int(getattr(args, "seed")) + shard_id * off)
    except Exception:
        pass

    if getattr(args, "outdir", ""):
        args.outdir = os.path.join(args.outdir, "shards", f"shard_{shard_id:03d}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True, allow_abbrev=False)
    p.add_argument("--phase", choices=["mc", "circuit", "test", "iqae", "mlae", "screen", "reference", "validate", "mps_probe", "mps_batch"], default="mc")
    p.add_argument("--config", default=None, help="Model JSON, relative to the project root")
    p.add_argument("--suite", default=None, help="Finite suite JSON, relative to the project root")
    p.add_argument('--case-id', default=None, help='Select one case before overriding a multi-case suite')
    p.add_argument('--counts-check', choices=('0','1'), default=None, help='MPS measurement mapping check; 0 skips the repeated measured A')
    p.add_argument('--mps-lapack', choices=('0','1'), default=None)
    p.add_argument('--probe-compile',action='store_true',help='circuit only: build/compile the MPS probe u/cx identity (no evolution)')
    p.add_argument('--blas-threads',type=int,default=1,help='MPS/IQAE: verified BLAS team, separate from MPS/OpenMP; default 1')
    p.add_argument('--perf-prefix-gates',type=int,default=None,help='mps_probe only: fixed strict prefix of the fully compiled A, performance scope only')
    p.add_argument('--perf-reference',default=None,help='same-prefix prior result for numerical equivalence and timing comparison')
    p.add_argument("--cpu-budget", type=int, default=4, help="Total CPU budget; one worker by default")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--histories", type=int, default=200000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--unitary-matched-energy", action="store_true")
    p.add_argument("--dir-tape", action="store_true",
                   help="Enable per-step direction tape (unitary embedding of classical direction resampling).")
    p.add_argument("--absorb-tape", action="store_true",
                   help="Enable per-step absorption coin tape (cA[t]) to better match classical MC semantics (uses +steps qubits).")
    p.add_argument("--term-tape", action="store_true",
                   help="Enable per-step termination scratch tapes for boundary/detect/absorb writes to avoid term reuse pollution (debug/alignment; uses +2*steps qubits).")
    p.add_argument("--scatter-tape", action="store_true",
                   help="Enable a per-step scatter-event tape tS[t] so scatter histories remain explicitly labeled through the end of the circuit.")
    p.add_argument("--full-history-mode", action="store_true",
                   help="Closer-to-classical stochastic embedding: force dir/absorb/term/scatter tapes on and disable aggressive tape cleanup/compaction.")
    p.add_argument("--kernel-profile", type=str, default="",
                   help="Production baseline by default; historical compressed/full_history retain their explicit circuit meaning.")
    p.add_argument("--test-measure-x", action="store_true",
                   help="In phase=test, also measure final x register and output x histogram.")
    p.add_argument("--test-measure-coins", action="store_true",
                   help="(Phase=test) Also measure interaction/absorption coin tapes (cI[t], cA[t]) and report per-step marginals + consistency checks. Use with --absorb-tape for cA.")
    p.add_argument("--test-measure-dir", action="store_true",
                   help="(Phase=test) Also measure final dir register and report invalid-dir rate (dir in {6,7} should be 0).")
    p.add_argument("--aer-shot-chunk", type=int, default=100,
                   help="(Phase=test) Split Aer sampling into chunks to provide progress. 0 disables chunking.")
    p.add_argument("--aer-heartbeat-sec", type=float, default=60.0,
                   help="Supervisor heartbeat interval in seconds; 0 disables heartbeats, not progress logs.")
    p.add_argument('--resource-sample-sec',type=float,default=1.)
    p.add_argument('--poll-sec',type=float,default=.1)
    p.add_argument('--debug-runtime',action=argparse.BooleanOptionalAction,default=False)

    p.add_argument("--compiled-qpy", type=str, default="",
                   help="(Phase=test) Load a pre-transpiled circuit from QPY and skip transpile.")
    p.add_argument("--save-compiled-qpy", type=str, default="",
                   help="(Phase=test) Save the transpiled circuit to this QPY path.")
    p.add_argument("--compile-only", action="store_true",
                   help="(Phase=test) Only transpile (and optionally save QPY) then exit.")
    p.add_argument("--skip-transpile", action="store_true",
                   help="(Phase=test) Run without transpile (advanced/debug; not recommended unless the circuit is already in safe basis).")

    p.add_argument(
        "--xt-no-interaction",
        action="store_true",
        help="[DEBUG] Force p_int=0 for all materials (MC and quantum) to test pure transport without collisions.",
    )
    p.add_argument(
        "--xt-disable-detector",
        action="store_true",
        help="[DEBUG] Disable detector ROI check in the quantum circuit (TEST/IQAE only).",
    )
    p.add_argument(
        "--xt-disable-boundary",
        action="store_true",
        help="[DEBUG] Disable boundary check in the quantum circuit (TEST/IQAE only).",
    )
    p.add_argument(
        "--debug-measure-work",
        action="store_true",
        help="[DEBUG] In TEST, also measure work ancillas and report whether they are cleaned (should be 0).",
    )
    p.add_argument("--plot", action="store_true", help="Save Phase-1 figures as PNG")
    p.add_argument("--outdir", type=str, default="", help="Output directory (default: outputs/run_<phase>_<timestamp>)")
    p.add_argument("--shots", type=int, default=20000)
    p.add_argument("--run-id", default="")
    p.add_argument("--request-hash", default="", help=argparse.SUPPRESS)
    p.add_argument("--root-seed", type=int, default=None)
    p.add_argument("--total-samples", type=int, default=None)
    p.add_argument("--transpiler-seed", type=int, default=1234)
    p.add_argument("--run-timeout-sec", "--overall-timeout-sec", dest="run_timeout_sec", type=float, default=None,
                   help="Optional positive wall-time limit. Omitted means no local deadline; inherited deadlines still apply.")
    p.add_argument("--run-memory-mib", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and record identity without building or simulating a circuit.")
    p.add_argument('--execute', dest='dry_run', action='store_false', help='Explicitly override CMD DRY_RUN=1')
    p.add_argument("--internal-worker", default="", help=argparse.SUPPRESS)

    p.add_argument("--shard-id", type=int, default=-1,
                   help="Shard index (>=0 enables shard mode; results go to <outdir>/shards/shard_XXX).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total shards (for metadata only).")
    p.add_argument("--shard-seed-offset", type=int, default=0,
                   help="If >0, auto-set seed = seed + shard_id * shard_seed_offset.")
    p.add_argument("--global-shot-offset", type=int, default=0,
                   help="Absolute starting shot index of this shard in the global run. Used only for layout-invariant chunk seeding in phase=test.")

    p.add_argument("--aer-threads", type=int, default=1,
                   help="Aer parallel threads (set 1 on Windows to reduce native crashes).")
    p.add_argument("--mps-threads", type=int, default=None,
                   help="Explicit Aer MPS OpenMP threads; omitted retains the recorded backend default.")
    p.add_argument("--aer-max-mem-mb", type=int, default=0,
                   help="If >0, cap Aer max_memory_mb (helps avoid OOM kill).")
    p.add_argument("--mps-max-bond", type=int, default=None,
                   help="Set backend bond explicitly, including 0; omitted preserves recorded backend default.")
    p.add_argument("--mps-trunc", type=float, default=None,
                   help="Set backend truncation explicitly, including exact 0; omitted preserves recorded backend default.")
    p.add_argument("--aer-safe-basis", action="store_true",
                   help="Compile circuits to a conservative basis [rz,sx,x,cx] (more stable, deeper).")

    p.add_argument('--iqae-mode', choices=['relative','absolute'], default='relative',
                   help='Formal relative precision by default; absolute is explicit historical compatibility.')
    p.add_argument('--iqae-diagnostic',choices=['compile','probability','sparse'],default=None,
                   help='Fixed-power diagnostic only; never invokes the IQAE estimator.')
    p.add_argument('--iqae-powers',default=None,help='Explicit finite list, e.g. 0,6 or 64.')
    p.add_argument('--iqae-max-sparse-states',type=int,default=60000)
    p.add_argument('--iqae-max-compiled-gates',type=int,default=4000000)
    p.add_argument('--iqae-compare-to',default=None,help='Existing diagnostic export directory for exact QPY comparison.')
    p.add_argument('--relative-error', type=float, default=.01, help='IQAE relative error target, independent of MPS bond/truncation.')
    p.add_argument('--confidence', type=float, default=.95, help='Confidence for the entire single-case IQAE process.')
    p.add_argument('--iqae-max-shots', type=int, default=200000, help='Total IQAE shot budget; no automatic extension.')
    p.add_argument('--iqae-max-queries', type=int, default=100000000000, help='Total A plus A-inverse call budget sum N*(2*k+1).')
    p.add_argument("--iqae-eps", type=float, default=0.05, help="Absolute epsilon, only with --iqae-mode absolute")
    p.add_argument("--iqae-alpha", type=float, default=0.05, help="Historical absolute-mode failure probability, only with --iqae-mode absolute")
    p.add_argument("--iqae-good", choices=["detect", "absorb", "boundary", "survive", "rare_detect"], default="detect",
                   help="Good state definition for IQAE (default: detect)")
    p.add_argument("--rare-eta", type=float, default=-1.0,
                   help="Rare-response efficiency eta for --iqae-good rare_detect.")
    p.add_argument("--rare-p-target", type=float, default=0.0,
                   help="Optional rare target probability recorded in output metadata.")
    p.add_argument("--iqae-max-grover-power", type=int, default=10000000,
                   help="Requested cap for IQAE Grover power. Enforced before submitting any power; 0 allows only k=0.")
    p.add_argument("--iqae-target-g", type=int, default=-1,
                   help="If >=0, estimate P(good & g==target).")
    p.add_argument("--iqae-use-aer", action="store_true", default=True,
                   help="Use Aer(MPS) sampler (recommended; default: True)")

    p.add_argument("--iqae-debug", action="store_true", default=False,
                   help="Enable verbose IQAE sampler debug logs (per-call timing / heartbeat).")
    p.add_argument("--iqae-heartbeat-sec", type=float, default=30.0,
                   help="AE supervisor heartbeat interval in seconds. 0 disables heartbeats.")
    p.add_argument("--iqae-shot-chunk", type=int, default=100,
                   help="(Phase=iqae) Split Aer sampling per sampler call into chunks for progress. 0 disables chunking.")
    p.add_argument("--iqae-max-call-sec", type=float, default=None,
                   help="Optional total AE worker wall-time budget, including build/compile/sampling. Omitted or 0 is unlimited.")
    p.add_argument("--iqae-debug-print-ops", action="store_true", default=False,
                   help="When --iqae-debug is on, also print top gate counts (can be slow on very large circuits).")
    p.add_argument("--mlae-powers", type=str, default="0,1",
                   help="Comma-separated fixed-depth AE powers for --phase mlae, e.g. 0,1 or 0,1,2.")

    p.add_argument("--epsilon", type=float, dest="iqae_eps", help=argparse.SUPPRESS)
    p.add_argument("--alpha", type=float, dest="iqae_alpha", help=argparse.SUPPRESS)
    p.add_argument("--g-target", type=int, dest="iqae_target_g", help=argparse.SUPPRESS)

    return p

REQUEST_FIELDS = {'phase':'task', 'config':'config', 'suite':'suite', 'case_id':'case_id',
    'steps':'steps', 'kernel_profile':'profile', 'mps_max_bond':'bond', 'mps_trunc':'mps_trunc',
    'cpu_budget':'cpu_budget', 'aer_threads':'aer_threads', 'mps_threads':'mps_threads',
    'run_memory_mib':'memory_mib', 'counts_check':'counts_check', 'shots':'shots', 'histories':'histories',
    'seed':'seed', 'outdir':'outdir', 'dry_run':'dry_run', 'mps_lapack':'mps_lapack',
    'blas_threads':'blas_threads','perf_prefix_gates':'perf_prefix_gates','perf_reference':'perf_reference',
    'relative_error':'relative_error','confidence':'confidence','iqae_max_shots':'iqae_max_shots',
    'iqae_max_queries':'iqae_max_queries','iqae_max_grover_power':'iqae_max_grover_power'}
REQUEST_FIELDS.update({key:key for key in ('iqae_diagnostic','iqae_powers',
    'iqae_max_sparse_states','iqae_max_compiled_gates','iqae_compare_to')})


def _resolve_request(args, original):
    from run_request import resolve_run_request
    parser = build_argparser()
    specified = {parser._option_string_actions[token.split('=',1)[0]].dest
                 for token in original if token.split('=',1)[0] in parser._option_string_actions}
    if args.phase == 'iqae':
        if args.iqae_mode == 'relative' and specified & {'iqae_eps','iqae_alpha'}:
            raise ValueError('Absolute epsilon/alpha requires --iqae-mode absolute; use --relative-error/--confidence for formal IQAE')
        if args.iqae_mode == 'absolute' and specified & {'relative_error','confidence'}:
            raise ValueError('Relative error/confidence cannot be combined with explicit historical absolute mode')
    if args.phase in REVISION_PHASES:
        allowed=set(REQUEST_FIELDS)|{'run_id','internal_worker','aer_heartbeat_sec','run_timeout_sec',
                                    'resource_sample_sec','poll_sec','debug_runtime'}
        unsupported=specified-allowed
        if unsupported:
            raise ValueError(f'Parameters inapplicable to {args.phase}: {sorted(unsupported)}')
    cli = {key:getattr(args,attr) for attr,key in REQUEST_FIELDS.items() if attr in specified}
    cmd = json.loads(os.environ.get('VRQO_CMD_REQUEST_JSON', '{}'))
    manual_flags=('dir_tape','absorb_tape','term_tape','scatter_tape','full_history_mode')
    if any(getattr(args,key,False) for key in manual_flags):
        if args.phase not in ('circuit','test','iqae','mlae'):
            raise ValueError('Manual quantum history flags are inapplicable to this phase')
        profile=cli.get('profile') or cmd.get('profile')
        if profile and profile!='manual':
            raise ValueError('Manual history flags cannot be silently ignored by a named profile; use --kernel-profile manual')
        cli['profile']='manual'
    resolved = resolve_run_request({'cli':cli, 'cmd':cmd}, project_root=PROJECT_ROOT)
    if args.phase == 'iqae' and args.iqae_mode == 'absolute':
        resolved['precision_contract'] = {
            'mode':'historical_absolute', 'absolute_epsilon':args.iqae_eps,
            'failure_probability':args.iqae_alpha,
            'formal_relative_precision_certified':False,
            'scope':'explicit historical compatibility; no 1% relative claim'}
    if cli.get('profile')=='manual':
        resolved['manual_profile_flags']={key:getattr(args,key,False) for key in manual_flags}
    for attr,key in REQUEST_FIELDS.items():
        if key not in resolved['arguments']:
            continue
        value = resolved['arguments'][key]
        if value is not None: setattr(args,attr,value)
    # The selected suite owns case steps/profiles, never argparse defaults.
    args.resolved_request = resolved
    return resolved


def _record_request(args):
    record = args.resolved_request
    write_json_new(Path(args.outdir)/'resolved_request.json', record)
    if record['effective_suite'] is not None:
        path = Path(args.outdir)/'resolved_suite.json'
        write_json_new(path, record['effective_suite'])
        args.suite = str(path)
    print(json.dumps({'resolved':str(Path(args.outdir)/'resolved_request.json'),
        'task':args.phase, 'cases':[{k:c.get(k) for k in ('id','config','steps','profile','profiles')} for c in record['cases']],
        'backend':record['backend'], 'resources':record['resources'], 'dry_run':args.dry_run}, ensure_ascii=False),flush=True)


def main(argv: Any = None) -> int:
    original=list(sys.argv[1:] if argv is None else argv)
    # Independent jobs share this entry point. The batch coordinator owns its
    # resource parser; each child still executes the ordinary mps_probe phase.
    selector=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    selector.add_argument('--phase')
    selected,_=selector.parse_known_args(original)
    if selected.phase=='mps_batch':
        from batch_runtime import main as batch_main
        return batch_main(original)
    args = build_argparser().parse_args(original)
    if args.probe_compile and (args.phase!='circuit' or args.aer_safe_basis):
        raise ValueError('--probe-compile requires phase=circuit and cannot combine with --aer-safe-basis')
    if os.environ.get('VRQO_DEADLINE_MONOTONIC'):
        raise ValueError('Inherited VRQO_DEADLINE_MONOTONIC is forbidden; clear it before scientific execution')
    _resolve_request(args, original)
    _configure_cpu_budget(args)
    if args.phase in ('screen', 'validate', 'mps_probe') and args.resolved_request['effective_suite'] is None:
        raise ValueError('--config or --suite is required for this phase')

    if type(args.steps) is not int or args.steps <= 0:
        print("steps must be positive")
        return 2

    if not args.outdir:
        args.outdir = _default_run_outdir(args.phase)
    args.outdir=str(validate_output_path(args.outdir))
    args.run_id=args.run_id or new_run_id()
    if args.seed<0 or args.shots<=0 or args.histories<=0 or args.global_shot_offset<0 or args.num_shards<=0:
        raise ValueError('Invalid sample count, seed, shard count or sample offset')
    if args.shard_seed_offset:raise ValueError('Implicit shard-seed offsets are unsupported; use a recorded runner plan')
    if (args.run_timeout_sec is not None and (not np.isfinite(args.run_timeout_sec) or args.run_timeout_sec<=0)) or (args.run_memory_mib is not None and args.run_memory_mib<=0):
        raise ValueError('Positive run time/memory limits required')
    for interval in (args.aer_heartbeat_sec,args.iqae_heartbeat_sec):
        if not np.isfinite(interval) or interval<0:raise ValueError('Heartbeat interval must be finite and nonnegative')
    for interval in (args.resource_sample_sec,args.poll_sec):
        if not np.isfinite(interval) or interval<=0:raise ValueError('Resource and polling intervals must be finite and positive')
    _sampler_timeout(args)
    _iqae_backend_options(args)
    cfg=_apply_xt_overrides(_load_config(args),args)
    if args.phase not in ('mc', *REVISION_PHASES):
        from config import validate_quantum_config
        validate_quantum_config(cfg,args.steps)
        _resolved_kernel_profile(args)
    for path in (args.compiled_qpy,args.save_compiled_qpy):
        if path:validate_output_path(path)
    if args.save_compiled_qpy:
        assert_new_file(args.save_compiled_qpy);assert_new_file(args.save_compiled_qpy+'.meta.json')
    if args.dry_run:
        # A parameter check creates no run directory; the same OUTDIR can then
        # be used for the one real execution after DRY_RUN is switched to zero.
        print(json.dumps({'status':'validated_only','request':args.resolved_request,
            'resources':args.resource_settings,'simulation_executed':False},ensure_ascii=False),flush=True)
        return 0
    # All computation starts inside an owned worker; dry-run remains read-only validation.
    if args.phase in ('mc','circuit','test','iqae','mlae',*REVISION_PHASES):
        expected=os.environ.get('VRQO_MAIN_WORKER_TOKEN')
        if args.internal_worker and args.internal_worker!=expected:raise ValueError('Invalid supervised-worker token')
        if not args.internal_worker:
            from supervised import run_limited
            token=new_run_id()
            run_destination=Path(args.outdir)
            if args.shard_id>=0:run_destination=run_destination/'shards'/f'shard_{args.shard_id:03d}'
            prepare_run_directory(run_destination)
            supervision=validate_output_path(run_destination/'_supervision')
            command=[sys.executable,'-B',str(Path(__file__).resolve()),*original,'--outdir',args.outdir,
                     '--run-id',args.run_id,'--internal-worker',token]
            result=run_limited(command,cwd=Path(__file__).resolve().parent,output_dir=supervision,
                timeout_sec=args.run_timeout_sec,memory_bytes=None if args.run_memory_mib is None else args.run_memory_mib*1024**2,
                threads=args.aer_threads,blas_threads=args.blas_threads,
                env_extra={'VRQO_MAIN_WORKER_TOKEN':token},echo_output=True,
                heartbeat_sec=args.iqae_heartbeat_sec if args.phase in ('iqae','mlae') else args.aer_heartbeat_sec,
                heartbeat_tag=f'{args.phase} shard={args.shard_id}',resource_sample_sec=args.resource_sample_sec,
                poll_sec=args.poll_sec,debug=args.debug_runtime)
            write_json_new(run_destination/'supervision_result.json',{
                **result,'process_result_path':str(supervision/'process_result.json'),
                'worker_results_path':str(run_destination/'results.json')})
            if args.debug_runtime:print(json.dumps({'supervision':str(supervision),**result},ensure_ascii=False))
            else:print(f'[END] phase={args.phase} status={result["status"]}',flush=True)
            return 0 if result['status']=='passed' else 3
    _apply_sharding(args)
    prepare_run_directory(args.outdir,
        reserved_subdirectories=('_supervision',) if args.internal_worker else ())
    _record_request(args)
    write_json_new(Path(args.outdir)/'resolved_config.json', {'config':normalize(cfg) if args.resolved_request['effective_suite'] is None else None,'steps':args.steps,
        'model_document':__import__('models').load_document(args.config) if args.config else None,
        'arguments':vars(args),'actual_cases':args.resolved_request['cases'],'precedence':args.resolved_request['precedence'],
        'suite_controls_cases':args.phase in ('screen','validate','mps_probe')})
    log_path=os.path.join(args.outdir,'run.log')
    # Batch workers already write one coordinator-owned raw log; do not copy it.
    with (nullcontext() if os.environ.get('VRQO_WORKER_STARTED_FILE') else _tee_to_file(log_path)):
        # Required startup acknowledgement is separate from best-effort display.
        # The batch counts a worker only after preflight and identity registration.
        started_file=os.environ.get('VRQO_WORKER_STARTED_FILE')
        if started_file:
            write_json_new(started_file,{'pid':os.getpid(),'run_id':args.run_id,
                'shard_id':args.shard_id,'request_hash':args.request_hash})
        supervised.publish_progress('worker_start',phase=args.phase)
        if args.phase in REVISION_PHASES:
            try:
                return phase_revision(args)
            except Exception as exc:
                write_json_new(Path(args.outdir)/'failure.json', {'status':'failed','type':type(exc).__name__,
                    'message':str(exc),'traceback':traceback.format_exc()})
                raise
        return {'mc':phase_mc,'circuit':phase_circuit,'test':phase_test,'iqae':phase_iqae,'mlae':phase_mlae}[args.phase](args)

if __name__ == "__main__":
    raise SystemExit(main())
