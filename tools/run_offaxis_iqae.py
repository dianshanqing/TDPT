"""Run a prospectively budgeted IQAE experiment on the off-axis five-step duct."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iqae import OracleSpec, build_grover_operator_for_spec, estimate_relative_iqae
from models import load_model
from reference_engine import build_traced, evolve, registers
from runtime_identity import circuit_fingerprint, software_versions
from tools.verify_transport_channels import end_checks, grover_step, mass


class BudgetStop(RuntimeError):
    status = "budget_limited"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory below outputs/")
    parser.add_argument("--pilot-seconds", type=float, required=True,
                        help="Measured seconds for one Q in this model's separate gate validation")
    args = parser.parse_args()
    if not math.isfinite(args.pilot_seconds) or args.pilot_seconds <= 0:
        parser.error("--pilot-seconds must be positive and finite")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Output must be below outputs/")
    out.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        with (out / name).open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")

    protocol_path = ROOT / "configs/offaxis_iqae_budget.json"
    p = json.loads(protocol_path.read_bytes())
    cfg = load_model(str(ROOT / p["model"]))
    transport, _, _ = build_traced(cfg, p["steps"], p["profile"])
    regs = registers(transport)
    Q, obj, _ = build_grover_operator_for_spec(
        A_circuit=transport, status_qubits=regs["status"], g_qubits=regs["g"],
        spec=OracleSpec(good=p["good"]))
    marked = Q.state_preparation
    if list(Q.reflection_qubits) != list(range(marked.num_qubits)) or len(obj) != 1:
        raise ValueError("Unexpected marking or zero-reflection register")
    wm = sum(1 << b for b in regs["work"])
    fm = 1 << obj[0]
    sm = sum(1 << b for b in regs["status"])
    detect = 1 << regs["status"][0]
    rng = np.random.Generator(np.random.PCG64(p["seed"]))
    save("protocol.json", {"parameters":p, "pilot_seconds_per_Q":args.pilot_seconds,
                           "UTC":datetime.now(timezone.utc).isoformat(),
                           "numpy_bit_generator":"PCG64", "software":software_versions(),
                           "fingerprints": {"U":circuit_fingerprint(transport),
                                            "marked_A":circuit_fingerprint(marked),
                                            "Q":circuit_fingerprint(Q)},
                           "input_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in (p["model"], "configs/offaxis_iqae_budget.json",
                                             "tools/run_offaxis_iqae.py", "tools/verify_transport_channels.py",
                                             "iqae.py","reference_engine.py")}})
    start = time.monotonic()
    checks = []
    shots = []
    cache = {}
    state = None
    highest = -1
    worst_seconds = args.pilot_seconds
    total_discard = 0.0
    pending = {}

    def admit(k):
        elapsed = time.monotonic() - start
        available = psutil.virtual_memory().available / (1024**2)
        projection = (k - max(highest, 0)) * worst_seconds * p["projection_safety_factor"]
        pending.update(requested_power=k, last_computed_power=highest,
                       elapsed_seconds=elapsed, available_MiB=available,
                       projected_additional_seconds=projection)
        if available < p["minimum_available_MiB_before_extension"]:
            raise BudgetStop("minimum available physical memory")
        if elapsed + projection > p["extension_admission_seconds"]:
            raise BudgetStop("projected power exceeds the remaining admission window")

    def sample(k, n):
        nonlocal state, highest, worst_seconds, total_discard
        if k not in cache:
            if highest > k:
                raise RuntimeError("Power selector requested an uncached lower power")
            admit(k)
            if state is None:
                state, prep = evolve(marked, max_states=p["max_basis_states"])
                highest = 0
                save("preparation.json", prep)
            while highest < k:
                admit(k)
                state, details = grover_step(Q, state, p["amplitude_cutoff"], p["max_basis_states"])
                save(f"Q_{highest+1:04d}.json", details)
                if details["status"] != "complete":
                    raise BudgetStop(details["status"])
                highest += 1
                total_discard += details["discarded_l2_sum"]
                worst_seconds = max(worst_seconds, details["seconds"])
            end = end_checks(state, wm)
            mismatch = mass(state, lambda b: bool(b & fm) != ((b & sm) == detect))
            probability = mass(state, lambda b: b & fm)
            bound = total_discard * (2*math.sqrt(end["norm_squared"]) + total_discard)
            if not 0 <= probability <= 1 or not all(
                value + bound < p["absolute_tolerance"] for value in
                (end["norm_residual"], end["work_nonzero_probability"], mismatch)
            ):
                raise ArithmeticError("Sparse simulation endpoint failed its error checks")
            checks.append({"power":k, "probability":probability, **end,
                           "marker_mismatch_probability":mismatch,
                           "cumulative_discard_l2_bound":total_discard,
                           "probability_error_bound":bound})
            cache[k] = probability
        hit = int(rng.binomial(n, cache[k]))
        shots.append({"power":k,"shots":n,"hits":hit,"native_gate_probability":cache[k]})
        save(f"batch_{len(shots):02d}.json", shots[-1])
        return hit

    estimate = estimate_relative_iqae(sample, shots=p["shots_per_round"],
              relative_error=p["relative_error"], confidence=p["confidence"],
              max_shots=p["max_shots"], max_queries=p["max_queries"], max_power=p["max_power"])
    save("result.json", {"estimate":estimate, "endpoint_checks":checks,
          "batches":shots,"last_extension_decision":pending,"highest_completed_prefix":highest,
          "elapsed_seconds":time.monotonic()-start})
    print(estimate["status"], estimate["reason"], estimate["costs"], estimate["last_valid_interval"])
    return 0 if estimate["status"] in ("completed", "budget_limited") else 1


if __name__ == "__main__":
    raise SystemExit(main())
