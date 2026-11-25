"""Replay the archived off-axis, event-history and bond-dimension comparisons."""
import csv
import gzip
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iqae import estimate_relative_iqae
import numpy as np
from scipy.stats import beta

def read(p):
    return json.loads(p.read_bytes())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_fingerprints(paths):
    for name, expected in paths.items():
        assert sha(ROOT / name) == expected, name

def verify_joints(rows):
    return max(abs(row[-2]-row[-1]) for row in rows)

def main():
    h = ROOT / "outputs/circuit_resources/history_controls/qlegacy_T3_tau23_T15"
    verify_fingerprints(read(h/"protocol.json")["sha256"])
    with (h/"slab_resources.csv").open(newline="") as f:
        data = {r["profile"]:r for r in csv.DictReader(f)}
    identities = read(h/"slab_compiled_fingerprints.json")
    assert len(data) == len(identities) == 5
    err = 0.0
    for profile,row in data.items():
        v = read(h / ("short_"+profile+".json"))
        assert v["passed"] and v["steps"] == len(v["forward"]["checkpoints"]) == 3
        assert len(v["register_map"]["work"]) == 17
        for step in v["forward"]["checkpoints"]:
            o = step["observables"]
            diff = verify_joints(o["joint_rows"])
            err = max(diff, err)
            assert diff == o["max_joint_error"] < 2e-11
            assert o["work_nonzero"] == 0 and abs(o["norm"]-1) < 2e-11
            assert o.get("scatter_ledger_mismatch_mass",0) == 0
        assert v["inverse_nonzero_probability"] < 2e-20 and v["inverse_zero_amplitude_error"] < 2e-11
        assert int(row["compiled_u"]) + int(row["compiled_cx"]) == int(row["compiled_gates"])
        item = identities[profile]
        if "archive" in item:
            assert sha(ROOT/item["archive"]) == item["archive_file_sha256"]
            assert item["compiled_fingerprint"] == item["archived_compiled_fingerprint"]
    print("History profiles:",len(data),"maximum stepwise joint error:",err)

    o = ROOT/"outputs/air_duct/offaxis_T5"
    r = read(o/"path_reference/path_probabilities.json")
    assert sha(ROOT/"configs/models/q_duct_offaxis.json") == r["config_sha256"]
    assert sha(ROOT/"tools/check_offaxis_paths.py") == r["driver_sha256"]
    pr = Decimal(r["p_detect_through_step_5"])
    with localcontext() as ctx:
        ctx.prec=50
        weights = [Decimal(x["detection_probability"]) for x in r["paths"]]
        assert abs(sum(weights)-pr) < Decimal("1e-48") and len(weights) == 12
        assert sum(x>0 for x in weights) == 6
        assert min(path["turns_from_source_heading"] for path,w in zip(r["paths"],weights) if w>0)==3
    c = read(o/"gate_02/classical.json")["rows"]
    assert [x["p_D"] for x in c[:-1]] == [0.,0.,0.,0.]
    assert abs(float(pr)-c[-1]["p_D"]) < 1e-18
    g = read(o/"gate_02/grover.json")
    a = read(o/"gate_02/preparation.json")
    ident = read(o/"gate_02/identity.json")
    verify_fingerprints(ident["source_sha256"])
    assert read(o/"gate_01/grover.json")["evolution"]["status"] == "basis_budget_exceeded"
    for checkpoint in a["evolution"]["checkpoints"]:
        obs = checkpoint["observables"]
        assert verify_joints(obs["joint"]) == obs["joint_max_abs_error"] < 2e-11
        assert obs["work_nonzero_probability"] == 0
    for row in (a["k0"], g["k1"]):
        ref = math.sin((2*row["k"]+1)*math.asin(math.sqrt(float(pr))))**2
        err = abs(row["flag_probability"]-ref)
        assert max(err,row["norm_residual"],row["work_nonzero_probability"],row["marker_mismatch_probability"]) + row["probability_error_bound_from_filter"] < 2e-11
    print("Off-axis direct checks: 5 step-end distributions and 2 marked readouts")

    q = o/"iqae_01"
    run = read(q/"result.json")
    agreement = read(q/"protocol.json")
    p = agreement["parameters"]
    assert p == read(ROOT/"configs/offaxis_iqae_budget.json")
    verify_fingerprints(agreement["input_sha256"])
    assert list(map(agreement["fingerprints"].get,["U","marked_A","Q"])) == list(map(ident["circuit_sha256"].get,["U_T","A_D","Q_D"]))
    rng = np.random.Generator(np.random.PCG64(p["seed"]))
    saved = run["batches"]
    for i,b in enumerate(saved,1):
        assert b == read(q/f"batch_{i:02d}.json")
        assert rng.binomial(b["shots"],b["native_gate_probability"]) == b["hits"]
        k = b["power"]
        direct = math.sin((2*k+1)*math.asin(math.sqrt(float(pr))))**2
        assert abs(direct-b["native_gate_probability"]) < 2e-11
    old = run["estimate"]
    used=0
    class ReplayStop(RuntimeError):
        status="budget_limited"
    def replay(k,n):
        nonlocal used
        if used == len(saved):
            assert k == run["last_extension_decision"]["requested_power"]
            raise ReplayStop("reached saved budget stop")
        b=saved[used]
        assert (k,n) == (b["power"], b["shots"])
        used += 1
        return b["hits"]
    result = estimate_relative_iqae(replay, relative_error=p["relative_error"],confidence=p["confidence"],
        shots=p["shots_per_round"],max_shots=p["max_shots"],max_queries=p["max_queries"],max_power=p["max_power"])
    for field in ["rounds","costs","last_valid_interval","status","precision_target_met"]:
        assert result[field] == old[field]
    for round in old["rounds"]:
        a=round["alpha"]; k=round["success_count"]; n=round["shots"]
        interval = [0. if k==0 else beta.ppf(a/2,k,n-k+1), 1. if k==n else beta.isf(a/2,k+1,n-k)]
        assert list(interval) == round["measurement_interval"]
    stats=old["costs"]
    assert sum(b["shots"]*(2*b["power"]+1) for b in saved) == stats["queries_A_and_inverse"]
    print("IQAE replay:",old["status"],stats["shots"],stats["queries_A_and_inverse"],old["last_valid_interval"])

    directories = [
        ROOT/"outputs/deep_penetration/tau23/mps/baseline_tau23_T15_bond64_mem1800",
        ROOT/"outputs/deep_penetration/tau23/mps/baseline_tau23_T15_bond256_threads1_01",
    ]
    raw = [json.loads(gzip.decompress(next(d.glob("*trunc*.json.gz")).read_bytes())) for d in directories]
    for key in ["original_A_fingerprint","compiled_A_fingerprint","diagnostic_fingerprint",
                "config","register_map","reference"]:
        assert raw[0][key] == raw[1][key], key
    assert raw[0]["compiled_A_fingerprint"] == identities["baseline"]["compiled_fingerprint"]
    for bond,d in zip((64,256),raw):
        diff = [abs(x-y) for x,y in zip(d["probabilities"],d["reference"])]
        residual = abs(sum(d["probabilities"])-1)
        assert diff == d["absolute_errors"] and residual == abs(d["raw_probability_sum"]-1)
        assert d["execution_status"] == "passed" and d["strict_status"] == "failed"
        assert max(diff) >= 2e-11
        print("Baseline MPS",bond,"D",d["probabilities"][1],"max error",max(diff),"norm residual",residual)

    summary = read(h / "summary.json")
    assert summary["all_stock_archive_comparisons_passed"]
    assert not summary["simulation_executed_on_slab"]
    expected = {
        "baseline": (156, 93447, 29901, 57958),
        "compressed": (156, 95472, 30546, 58486),
        "baseline_scatter_copy": (171, 93462, 29916, 57959),
        "baseline_scatter_control": (171, 93462, 29916, 57314),
        "full_history": (171, 81537, 26481, 50564),
    }
    for profile, values in expected.items():
        assert tuple(int(data[profile][k]) for k in
                     ("qubits","compiled_gates","compiled_cx","compiled_depth")) == values

    mps = directories[1].parent
    audit = read(mps.parents[1] / "audits/precision_1pct_20261106/historical_reuse_audit.json")["rows"][3]
    assert audit["bond"] == 128 and audit["actual_profile"]["name"] == "compressed"
    cmp64 = read(next((mps/"scatter_tau23").glob("*trunc*.json")))
    cmp256 = json.loads(gzip.decompress(next((mps/"compressed_tau23_T15_bond256_01").glob("*trunc*.gz")).read_bytes()))
    for record, bond, channel, detection, largest in (
        (cmp64, 64, 2, 0.0, "5.6224e-08"),
        (audit, 128, 2, 0.0, "4.2984e-09"),
        (cmp256, 256, 3, 1.0728112e-10, "3.5138e-10"),
        (raw[0], 64, 2, 1.1847175e-36, "2.3326e-07"),
        (raw[1], 256, 3, 1.0728946e-10, "3.3185e-10"),
    ):
        assert record["bond"] == bond and record["reference"] == raw[1]["reference"]
        p_det = record["probabilities"][1]
        assert math.isclose(p_det, detection, rel_tol=5e-8, abs_tol=0)
        errors = [abs(x-y) for x,y in zip(record["probabilities"],record["reference"])]
        assert errors.index(max(errors)) == channel and format(max(errors), ".4e") == largest
    assert stats["shots"] == 1024 and stats["queries_A_and_inverse"] == 8192
    assert old["status"] == "budget_limited" and not old["precision_target_met"]
    print("Five compiled resource rows and five slab settings checked; bond 128 uses its historical audit.")


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run verification with assertions enabled")
    main()
