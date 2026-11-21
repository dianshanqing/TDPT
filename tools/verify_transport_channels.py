"""Sparse gate evolution of three terminal markers on the five-step reduced duct."""
import argparse
import hashlib
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iqae import OracleSpec, build_grover_operator_for_spec
from models import config_dict, load_model
from reference_engine import apply_instruction, build_traced, evolve, flattened, registers
from reference_solver import reference
from runtime_identity import assert_new_file, circuit_fingerprint, software_versions, write_json_new
from validation import TOL

INPUT = "configs/models/q_legacy_small.json"
PROFILE = "baseline"
STEPS = 5
CHANNELS = {"detect": 1, "absorb": 2, "boundary": 3}


def mass(state, predicate):
    return math.fsum(abs(a) ** 2 for basis, a in state.items() if predicate(basis))


def end_checks(state, work_mask):
    norm = math.fsum(abs(a) ** 2 for a in state.values())
    work = mass(state, lambda basis: basis & work_mask)
    return {"norm_squared": norm, "norm_residual": abs(norm - 1),
            "work_nonzero_probability": work}


def grover_step(circuit, state, cutoff, max_states):
    """Use the same primitive operations and matrices as reference_engine.evolve."""
    events = []
    peak = len(state)
    start = time.perf_counter()
    # An unconditional X relabels the sparse basis exactly. Carry the XOR
    # offset through X gates and materialize it before other gate matrices.
    frame = 0
    for number, gate in enumerate(flattened(circuit), 1):
        if gate[0] == "gate" and gate[1] == "x":
            _, name, control_mask, control_value, targets, matrix = gate
            if control_mask == 0:
                frame ^= 1 << targets[0]
                continue
            gate = ("gate", name, control_mask, control_value ^ (frame & control_mask), targets, matrix)
        elif frame:
            state = {basis ^ frame: a for basis, a in state.items()}
            frame = 0
        state = apply_instruction(state, gate)
        size = len(state)
        peak = max(peak, size)
        if size > max_states:
            return None, {"status": "basis_budget_exceeded", "primitive_index_1based": number,
                          "primitive": gate[1], "basis_size_before_filter": size,
                          "max_states": max_states, "seconds": time.perf_counter() - start}
        if cutoff and gate[0] == "gate" and gate[1] in ("h", "ry", "unitary"):
            tiny = [abs(a) for a in state.values() if abs(a) < cutoff]
            if tiny:
                events.append({"primitive_index_1based": number, "count": len(tiny),
                               "discarded_l2": math.hypot(*tiny)})
                state = {basis: a for basis, a in state.items() if abs(a) >= cutoff}
    if frame:
        state = {basis ^ frame: a for basis, a in state.items()}
    return state, {"status": "complete", "primitive_gates": number, "peak_basis": peak,
                   "final_basis": len(state), "discard_events": events,
                   "discarded_l2_sum": math.fsum(e["discarded_l2"] for e in events),
                   "seconds": time.perf_counter() - start}


def evaluate(cutoff, max_states, targets):
    cfg = load_model(INPUT)
    classical = reference(cfg, STEPS)
    transport, _, points = build_traced(cfg, STEPS, PROFILE)
    regs = {reg.name: reg for reg in transport.qregs}
    mapping = registers(transport)
    work_mask = sum(1 << bit for bit in mapping["work"])
    assert len(mapping["work"]) == 17 and len(points) == STEPS
    paths = [INPUT, "models.py", "presets.py", "config.py", "unitary_circuit.py", "iqae.py",
             "reference_engine.py", "reference_solver.py", "runtime_identity.py", "validation.py",
             "tools/verify_transport_channels.py"]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": "q_legacy_small",
        "profile": PROFILE, "steps": STEPS, "model_input": INPUT, "resolved_config": config_dict(cfg),
        "software_versions": software_versions(),
        "input_and_source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "unmarked_transport_sha256": circuit_fingerprint(transport), "register_map": mapping,
        "transport_qubits": transport.num_qubits, "probability_tolerance": TOL,
        "comparison": "absolute < tolerance", "max_states": max_states, "amplitude_cutoff": cutoff,
        "reference_probability_order": classical["order"], "classical_steps": classical["rows"],
        "channels": {}, "measurements": []
    }
    rows = result["measurements"]
    passed = all(row["norm_error"] < TOL for row in classical["rows"])
    for name in targets:
        status = CHANNELS[name]
        p = classical["rows"][-1]["probabilities"][status]
        Q, obj, _ = build_grover_operator_for_spec(A_circuit=transport, status_qubits=regs["status"],
                                                   g_qubits=regs["g"], spec=OracleSpec(good=name))
        marked = Q.state_preparation
        assert len(obj) == 1 and registers(marked)["flag"] == obj
        flag_mask = 1 << obj[0]
        reflection = list(Q.reflection_qubits)
        assert reflection == list(range(marked.num_qubits))
        psi, run = evolve(marked, max_states=max_states, checkpoints=points,
                          observer=lambda qc, state, point: end_checks(state, work_mask))
        checkpoint_steps = [c["step"] for c in run["checkpoints"]]
        if checkpoint_steps != list(range(STEPS)):
            raise RuntimeError(f"Expected five successive step ends; got {checkpoint_steps}")
        info = result["channels"][name] = {
            "status_code": format(status, "02b"), "objective_index": obj[0],
            "reflection_qubits": reflection, "marked_A_sha256": circuit_fingerprint(marked),
            "Q_sha256": circuit_fingerprint(Q), "marked_A_evolution": run,
        }
        passed &= all(c["observables"]["work_nonzero_probability"] < TOL
                      and c["observables"]["norm_residual"] < TOL for c in run["checkpoints"])
        for k in (0, 1):
            epsilon = 0.0
            if k == 1:
                psi, details = grover_step(Q, psi, cutoff, max_states)
                info["Q_evolution"] = details
                if psi is None:
                    passed = False
                    break
                epsilon = details["discarded_l2_sum"]
            checks = end_checks(psi, work_mask)
            observed = mass(psi, lambda basis: basis & flag_mask)
            expected = math.sin((2*k + 1)*math.asin(math.sqrt(p)))**2
            error = abs(observed - expected)
            bound = epsilon * (2 * math.sqrt(checks["norm_squared"]) + epsilon)
            status_bits = mapping["status"]
            marking_error = mass(psi, lambda basis:
                bool(basis & flag_mask) != (sum(((basis >> b) & 1) << i
                                                for i, b in enumerate(status_bits)) == status))
            row_ok = all(v + bound < TOL for v in (error, checks["norm_residual"],
                                                    checks["work_nonzero_probability"], marking_error))
            rows.append({"channel": name, "status_code": format(status, "02b"), "k": k,
                         "classical_probability": p, "observed_marked_probability": observed,
                         "expected_from_classical": expected, "absolute_error": error,
                         **checks, "status_marker_mismatch_probability": marking_error,
                         "probability_error_bound_from_filter": bound, "passed": row_ok})
            passed &= row_ok
        print(f"{name}: {info['Q_evolution']['status']}", flush=True)
    result["maximum_absolute_error_among_completed_rows"] = max(r["absolute_error"] for r in rows)
    result["completed_rows"] = len(rows)
    result["passed"] = bool(passed and len(rows) == len(targets) * 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New JSON file inside outputs/.")
    parser.add_argument("--cutoff", type=float, default=1e-17, help="Set zero for unfiltered evolution.")
    parser.add_argument("--max-states", type=int, default=3000000)
    parser.add_argument("--channel", choices=tuple(CHANNELS), help="Check a single channel.")
    args = parser.parse_args()
    if not math.isfinite(args.cutoff) or args.cutoff < 0 or args.max_states < 1:
        parser.error("The cutoff must be finite and nonnegative, and max-states positive.")
    target = assert_new_file(args.out)
    result = evaluate(args.cutoff, args.max_states, [args.channel] if args.channel else CHANNELS)
    write_json_new(target, result)
    print(f"{result['completed_rows']} rows, passed={result['passed']}, saved {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
