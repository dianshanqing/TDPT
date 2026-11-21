"""Rebuild and compile the three-step two-cell detector oracle and audit its records."""
import argparse
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from itertools import zip_longest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from qiskit import transpile
from qiskit_algorithms import IterativeAmplitudeEstimation
import iqae_sparse_complete as completed
from iqae import estimate_relative_iqae
from reference_engine import flattened, getval, registers
from runtime_identity import normalize, software_versions

BASE = ROOT / "outputs/amplitude_estimation/tau19_two_cell"
SUCCESS = BASE / "iqae_sparse_complete_20261117_124813_443bf938"
EARLY = BASE / "20261117_budget_limited"
TOL = 2e-11


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild(driver, batch, work):
    """Run each original worker through its identity checkpoint, before sampling."""
    saved = read(batch / "circuit_identity.json")
    protocol = read(batch / "protocol.json")
    require(saved["protocol"] == protocol, "Protocol differs between archived records")
    require(sha(Path(driver.__file__)) == protocol["driver_sha256"], "Driver fingerprint differs")
    for name, fingerprint in protocol["production_sources"].items():
        require(sha(ROOT / name) == fingerprint, "Frozen source differs: " + name)
    # Each worker verifies a freshly prepared local protocol with the local platform.
    if driver is completed:
        driver.prepared(work, driver.DEFAULT_PROTOCOL.copy())
    else:
        driver.prepared(work)
    local = read(work / "protocol.json")
    require(local["config"] == protocol["config"] and local["protocol"] == protocol["protocol"],
            "Physical or estimator input differs")
    circuits = {}

    class IdentityReady(Exception):
        pass

    def checkpoint(path, identity):
        require(path.name == "circuit_identity.json", "Unexpected worker output")
        frame = inspect.currentframe().f_back
        require(frame.f_code is driver.worker.__code__, "Expected the original worker")
        for key in ("A", "Q", "marked", "obj", "engine"):
            circuits[key] = frame.f_locals[key]
        circuits["identity"] = identity
        raise IdentityReady

    with patch.object(driver, "write_new", checkpoint):
        try:
            driver.worker(work)
        except IdentityReady:
            pass
    require(bool(circuits), "Worker stopped before building the oracle")
    local_id = circuits["identity"]["fingerprints"]
    for name in ("A", "marked_A"):
        require(local_id[name] == saved["fingerprints"][name], "Different preparation: " + name)
    require(saved["primitive_components"] == circuits["identity"]["primitive_components"], "Gate counts differ")
    return circuits, {"batch": batch.name, "driver_sha256": protocol["driver_sha256"],
                      "archived_versions": protocol["versions"], "archived": saved["fingerprints"],
                      "local": local_id, "matches_archive": {
                          k: v == saved["fingerprints"][k] for k, v in local_id.items()}}


def gate_transcript(q):
    # Open the Grover wrapper once, preserving the constructors' ordered operands.
    expanded = q.decompose(reps=1)
    types, gates, lookup = [], [], {}
    for inst in expanded.data:
        op = inst.operation
        base = getattr(op, "base_gate", op)
        entry = normalize({"name": op.name, "params": list(op.params),
                           "num_controls": getattr(op, "num_ctrl_qubits", 0),
                           "control_state": getattr(op, "ctrl_state", None),
                           "target_matrix": base.to_matrix()})
        key = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        if key not in lookup:
            lookup[key] = len(types)
            types.append(entry)
        gates.append([lookup[key], [expanded.find_bit(b).index for b in inst.qubits]])
    return {"register_bits": registers(expanded), "global_phase_rad": float(expanded.global_phase),
            "types": types, "gates_in_application_order": gates}


def matrix_check(q):
    """Independently expand -A S0 A† Sf in chronological order."""
    bits = list(range(q.num_qubits))
    require(list(q.reflection_qubits) == bits, "Incomplete zero reflection")
    prep = list(flattened(q.state_preparation))
    require(all(g[0] == "gate" for g in prep), "Unexpected independent phase in preparation")
    def primitive(name, targets, mat, controls=0):
        return ("gate", name, controls, controls, targets, np.asarray(mat, dtype=complex))
    x = [[0, 1], [1, 0]]
    h = np.asarray([[1, 1], [1, -1]]) / math.sqrt(2)
    end = bits[-1]
    flips = [primitive("x", [b], x) for b in bits]
    zero = flips + [primitive("h", [end], h), primitive("x", [end], x, (1 << end)-1),
                    primitive("h", [end], h)] + flips
    adjoint = [(op, name, cm, cv, target, m.conj().T)
               for op, name, cm, cv, target, m in reversed(prep)]
    expected = [primitive("z", [end], [[1, 0], [0, -1]])] + adjoint + zero + prep
    actual = list(flattened(q))
    require(actual[0][0] == "phase", "Missing Grover phase")
    phase_error = abs(actual[0][1] + 1)
    max_error = 0.0
    for j, (a, b) in enumerate(zip_longest(actual[1:], expected)):
        require(a is not None and b is not None and a[:5] == b[:5], f"Gate or wire differs at {j}")
        max_error = max(max_error, float(np.max(np.abs(a[5] - b[5]))))
    require(max(max_error, phase_error) < TOL, "Matrix or global phase mismatch")
    return {"primitive_gates_compared": len(expected), "phase_factor_difference_from_minus_one": phase_error,
            "largest_matrix_element_abs_difference": max_error, "zero_reflection_qubits": bits,
            "flag": end, "application_order": ["Sf", "marked_A_dagger", "S0", "marked_A"],
            "overall_phase_factor": -1, "primitive_counts_in_order": [1, len(adjoint), len(zero), len(prep)]}


def resources(circuit):
    compiled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0,
                         seed_transpiler=917, num_processes=1)
    counts = dict(compiled.count_ops())
    require(set(counts) <= {"u", "cx"} and compiled.layout is None, "Unexpected basis or mapping")
    return {"qubits": compiled.num_qubits, "total_gates": compiled.size(), "u": counts["u"],
            "cx": counts["cx"], "depth": compiled.depth()}


def histories(circuits, protocol, p_d):
    engine = circuits["engine"]
    state = engine.evolve(engine.Aops, {0: 1+0j})
    regs = registers(circuits["marked"])
    flag = circuits["obj"][0]
    require(regs["flag"] == [flag], "Marker bit mismatch")
    buckets = [[], []]
    residual = []
    for key, amp in state.items():
        if (key >> flag) & 1:
            require(getval(key, regs["status"]) == 1, "Incorrect detector marker")
            # Nonzero tA on a detected baseline path records a collision. Nonzero
            # dirT would mean at least one direction changed before detection.
            if getval(key, regs["dirT"]) or getval(key, regs["dir"]) or getval(key, regs["cA"]):
                residual.append(abs(amp)**2)
            else:
                buckets[bool(getval(key, regs["tA"]))].append(abs(amp)**2)
        else:
            require(getval(key, regs["status"]) != 1, "Unmarked detector branch")
    measured = [math.fsum(v) for v in buckets]
    geom = protocol["config"]["geom"]
    source = protocol["config"]["source"]
    n = geom["slab_x1"] - geom["slab_x0"] + 1
    require(n == 2 and geom["detector_x"] - source["x0"] == n and protocol["protocol"]["steps"] == n+1,
            "The straight-path analytic reference requires the fastest detection horizon")
    mat = protocol["config"]["mats"]["tungsten"]
    encounter, absorption = mat["p_int"][0], mat["p_abs"][0]
    free, forward = 1 - encounter, encounter*(1-absorption)/6
    reference = [free**2, 2*free*forward+forward**2]
    errors = [abs(a-b) for a,b in zip(measured, reference)]
    norm = math.fsum(abs(a)**2 for a in state.values())
    work = math.fsum(abs(a)**2 for k,a in state.items() if getval(k, regs["work"]))
    checks = errors + [math.fsum(residual), abs(math.fsum(measured)-p_d), abs(norm-1), work]
    require(all(0 <= d < TOL for d in checks), "History, mass or norm check failed")
    return {"unconditional_joint_events": ["detect_and_no_collision", "detect_and_forward_scatter_only"],
            "actual": measured, "analytic": reference, "abs_differences": errors,
            "other_marked_mass": math.fsum(residual), "joint_sum": math.fsum(measured),
            "reference_detector": p_d, "absolute_sum_error": abs(math.fsum(measured)-p_d),
            "raw_norm": norm, "work_nonzero_probability": work, "status": "passed"}


def replay(batch):
    record = read(batch / "result.json")
    p = record["protocol"]["protocol"]
    batches = iter(record["measurement_batches"])
    rng = np.random.default_rng(p["seed"])
    def observe(k, n):
        row = next(batches)
        require((k,n) == (row["power"],row["shots"]), "Unexpected measurement allocation")
        require(rng.binomial(n, row["actual_gate_probability"]) == row["success_count"],
                "Independent seeded draw differs")
        return row["success_count"]
    estimate = completed.estimate_with_protocol(observe, p)
    require(next(batches, None) is None and estimate == record["estimator"], "Estimator replay differs")
    selector = IterativeAmplitudeEstimation(epsilon_target=.1, alpha=estimate["alpha"])
    k, upper, theta = 0, True, [0., .25]
    for j,row in enumerate(estimate["rounds"],1):
        k, upper = selector._find_next_k(k, upper, theta)
        require(k == row["power"] and row["alpha"] == estimate["alpha"]/(j*(j+1)), "Risk/branch differs")
        theta = row["theta_interval"]
    next_k = None
    if estimate["status"] == "budget_limited":
        next_k = int(selector._find_next_k(k, upper, theta)[0])
        require(next_k > p["max_power"], "Budget stopping reason differs")
    return {"archived_batch":batch.name,"identical_estimator":True,"seeded_counts_reproduced":True,
            "status":estimate["status"],"rounds":[{k:r[k] for k in ("power","shots","success_count","alpha")}
            for r in estimate["rounds"]],"last_valid_interval":estimate["last_valid_interval"],
            "failure_probability_spent":estimate["failure_probability_spent"],"costs":estimate["costs"],
            "power_budget":p["max_power"],"rejected_next_power":next_k}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory for local results")
    out = parser.parse_args().out_dir
    if out.exists():
        parser.error("Choose a new output directory")
    # Import the historical module while keeping its machine-era default out of sys.path.
    spec = importlib.util.spec_from_file_location("budget_driver", EARLY / "driver_executed.py")
    earlier = importlib.util.module_from_spec(spec)
    saved_path = sys.path.copy()
    try:
        spec.loader.exec_module(earlier)
    finally:
        sys.path[:] = saved_path
    out.mkdir(parents=True, exist_ok=False)
    current, identity = rebuild(completed, SUCCESS, out / "complete-protocol")
    previous, early_id = rebuild(earlier, EARLY, out / "budget-protocol")
    # These disposable snapshots contain only the local reconstruction inputs.
    for scratch in (out / "complete-protocol", out / "budget-protocol"):
        (scratch / "protocol.json").unlink()
        scratch.rmdir()
    q = current["Q"]
    stream = gate_transcript(q)
    require(stream == gate_transcript(previous["Q"]), "Local driver reconstructions differ")
    require(identity["local"] == early_id["local"], "Local recursive fingerprints differ")
    gates = matrix_check(q)
    estimates = [replay(SUCCESS), replay(EARLY)]
    counts = iter([2,1])
    repeated = estimate_relative_iqae(lambda k,n: next(counts), shots=4, max_shots=8)
    rounds = [[r["power"], r["shots"], r["success_count"]] for r in repeated["rounds"]]
    require(rounds == [[0,4,2],[0,4,1]], "Repeat-power fresh batch check failed")
    a_cost, q_cost = resources(current["marked"]), resources(q)
    max_k = estimates[0]["costs"]["max_power"]
    result = {"model":"q_tau19_two_cell","steps":3,"profile":"baseline","objective":"detect",
              "package_version":"2.0.1","versions":software_versions(),"absolute_tolerance":TOL,
              "audit_tool_sha256":sha(Path(__file__)), "driver_reconstructions":[identity, early_id],
              "local_driver_gate_streams_identical":True,"operator_check":gates,
              "basis_gates":["u","cx"],"optimization_level":0,"seed_transpiler":917,
              "coupling_map":None,"backend_target":None,
              "local_transpilation":{"A_D":a_cost,"Q_D":q_cost,"largest_recorded_power":max_k,
                  "concatenated_depth_upper_bound":a_cost["depth"]+max_k*q_cost["depth"]},
              "replays":estimates,"synthetic_same_power_test_k_shots_hits":rounds,
              "path_classes":histories(current, read(SUCCESS/"protocol.json"),
                                       read(SUCCESS/"result.json")["posthoc_reference_p_D"])}
    data = json.dumps(stream, sort_keys=True, separators=(",",":"), allow_nan=False).encode()
    result["uncompressed_gate_transcript_sha256"] = hashlib.sha256(data).hexdigest()
    (out/"gate_stream.json.gz").write_bytes(gzip.compress(data,mtime=0))
    with (out/"result.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print("Local A_D, Q_D and depth bound:",result["local_transpilation"])


if __name__ == "__main__":
    main()
