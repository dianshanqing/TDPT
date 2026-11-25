"""Trace the five-step off-axis duct, then apply one actual detection Grover circuit."""
import argparse
import hashlib
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iqae import OracleSpec, build_grover_operator_for_spec
from models import config_dict, load_model
from reference_engine import build_traced, evolve, getval, marginal, registers
from reference_solver import reference
from runtime_identity import circuit_fingerprint, prepare_run_directory, software_versions, write_json_new
from validation import TOL
from verify_transport_channels import end_checks, grover_step, mass

INPUT = "configs/models/q_duct_offaxis.json"
STEPS = 5
JOINT = ["status", "x", "y", "z", "dir", "g"]
SOURCES = [INPUT, "config.py", "models.py", "unitary_circuit.py", "iqae.py",
           "reference_engine.py", "reference_solver.py", "validation.py", "runtime_identity.py",
           "tools/verify_transport_channels.py", "tools/verify_offaxis.py"]


def machine_memory():
    process = psutil.Process().memory_info()
    return {"available_physical_bytes": psutil.virtual_memory().available,
            "process_peak_working_set_bytes": getattr(process, "peak_wset", process.rss)}


def marker_readout(state, flag_bit, status_bits, work_mask, k, reference_probability, epsilon):
    checks = end_checks(state, work_mask)
    observed = mass(state, lambda b: (b >> flag_bit) & 1)
    mismatch = mass(state, lambda b: bool((b >> flag_bit) & 1) != (getval(b, status_bits) == 1))
    expected = math.sin((2 * k + 1) * math.asin(math.sqrt(reference_probability))) ** 2
    error = abs(observed - expected)
    bound = epsilon * (2 * math.sqrt(checks["norm_squared"]) + epsilon)
    return {"k": k, "flag_probability": observed, "expected_from_classical": expected,
            "absolute_error": error, "marker_mismatch_probability": mismatch, **checks,
            "probability_error_bound_from_filter": bound,
            "passed": max(error, checks["norm_residual"], checks["work_nonzero_probability"],
                          mismatch) + bound < TOL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Fresh directory under outputs/")
    parser.add_argument("--max-states", type=int, default=650000)
    parser.add_argument("--cutoff", type=float, default=1e-16)
    args = parser.parse_args()
    if args.max_states < 1 or not math.isfinite(args.cutoff) or args.cutoff < 0:
        parser.error("A positive state budget and a finite nonnegative cutoff are required.")
    out = prepare_run_directory(args.out)
    start = time.monotonic()
    memory_start = machine_memory()
    cfg = load_model(INPUT)
    classical = reference(cfg, STEPS)
    write_json_new(out / "classical.json", classical)
    p = classical["rows"][-1]["p_D"]
    transport, _, points = build_traced(cfg, STEPS, "baseline")
    regs = {reg.name: reg for reg in transport.qregs}
    mapping = registers(transport)
    assert len(mapping["work"]) == 17 and [point["step"] for point in points] == list(range(STEPS))
    work_mask = sum(1 << bit for bit in mapping["work"])
    Q, obj, _ = build_grover_operator_for_spec(
        A_circuit=transport, status_qubits=regs["status"], g_qubits=regs["g"],
        spec=OracleSpec(good="detect"))
    marked = Q.state_preparation
    assert registers(marked)["flag"] == obj and len(obj) == 1
    assert list(Q.reflection_qubits) == list(range(marked.num_qubits))

    def checkpoint(qc, state, point):
        step = point["step"] + 1
        target = reference(cfg, step, retain_joint=True)["joint"]
        measured = marginal(qc, state, JOINT)
        rows = [list(key) + [measured.get(key, 0.0), target.get(key, 0.0)]
                for key in sorted(measured.keys() | target.keys())]
        max_error = max(abs(row[-1] - row[-2]) for row in rows)
        checks = end_checks(state, work_mask)
        four = [math.fsum(row[-2] for row in rows if row[0] == status) for status in range(4)]
        status_error = max(abs(a-b) for a, b in zip(four, classical["rows"][step-1]["probabilities"]))
        return {"step": step, "joint": rows, "joint_max_abs_error": max_error,
                "status_probabilities": four, "status_max_abs_error": status_error, **checks,
                "passed": max(max_error, status_error, checks["norm_residual"],
                              checks["work_nonzero_probability"]) < TOL}

    state, prep = evolve(marked, checkpoints=points, observer=checkpoint, max_states=args.max_states)
    assert len(prep["checkpoints"]) == STEPS
    k0 = marker_readout(state, obj[0], mapping["status"], work_mask, 0, p, 0.0)
    passed_prep = k0["passed"] and all(c["observables"]["passed"] for c in prep["checkpoints"])
    identity = {"model": INPUT, "steps": STEPS, "profile": "baseline", "channel": "detect",
                "software": software_versions(), "resolved_config": config_dict(cfg),
                "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES},
                "circuit_sha256": {"U_T": circuit_fingerprint(transport),
                                   "A_D": circuit_fingerprint(marked), "Q_D": circuit_fingerprint(Q)},
                "register_map_including_marker": registers(marked), "reflection": list(Q.reflection_qubits),
                "transport_qubits": transport.num_qubits, "marked_qubits": marked.num_qubits,
                "max_states": args.max_states, "grover_amplitude_cutoff": args.cutoff,
                "comparison": "absolute error plus any filtering bound strictly below tolerance",
                "tolerance": TOL, "utc": datetime.now(timezone.utc).isoformat()}
    write_json_new(out / "identity.json", identity)
    write_json_new(out / "preparation.json", {
        "k0": k0, "evolution": prep, "classical_probability": p, "amplitude_cutoff": 0,
        "joint_columns": JOINT + ["gate_probability", "reference_probability"],
        "status_order": classical["order"], "memory_at_start": memory_start,
        "memory_after_preparation": machine_memory(), "passed": passed_prep})
    print(f"k=0 {k0['flag_probability']:.17g}, start Q using {len(state)} entries", flush=True)
    state, run = grover_step(Q, state, args.cutoff, args.max_states)
    k1 = None
    if state is not None:
        k1 = marker_readout(state, obj[0], mapping["status"], work_mask, 1, p,
                            run["discarded_l2_sum"])
    write_json_new(out / "grover.json", {"k1": k1, "evolution": run, "final_memory": machine_memory(),
                                        "wall_seconds": time.monotonic() - start,
                                        "passed": bool(k1 and k1["passed"])})
    print(run["status"], k1, flush=True)
    return 0 if passed_prep and k1 and k1["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
