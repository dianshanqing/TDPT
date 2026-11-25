"""Compare two direct-probability probes of the same tau23 baseline transport circuit."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


STATUS = ["survive", "detect", "absorb", "boundary"]
TOL = 2e-11
REFERENCE = "outputs/deep_penetration/tau23/classical/compact_n8_w0002_tau23.json"


def read_probe(path):
    packed = path.read_bytes()
    raw = gzip.decompress(packed) if path.suffix == ".gz" else packed
    row = json.loads(raw)
    if row["execution_status"] != "passed":
        raise ValueError(f"Execution did not return a state distribution in {path.name}.")
    if row["case"] != dict(config="compact_n8_w0002_tau23", id="n8_w0002_tau23_T15",
                            profile="baseline", steps=15):
        raise ValueError(f"Unexpected case in {path.name}.")
    if row["identity"]["measurement"]["status_order"] != STATUS:
        raise ValueError(f"Unexpected state order in {path.name}.")
    probs = row["probabilities"]
    ref = row["reference"]
    if len(probs) != 4 or len(ref) != 4 or not all(map(math.isfinite, probs + ref)):
        raise ValueError(f"Missing or invalid marginals in {path.name}.")
    errors = [abs(val - exact) for val, exact in zip(probs, ref)]
    residual = abs(math.fsum(probs) - 1)
    valid = all(0 <= x <= 1 for x in probs + ref) and abs(math.fsum(ref) - 1) < TOL
    passed = valid and max(errors) < TOL and residual < TOL
    rel = errors[1] / ref[1]
    if errors != row["absolute_errors"] or residual != row["normalization_residual"]:
        raise ValueError(f"Recorded numerical errors differ from the recomputed values in {path.name}.")
    if passed != row["passed_status_absolute_tolerance"] or rel != row["p_D_relative_error"]:
        raise ValueError(f"Strict status or detection error mismatch in {path.name}.")
    result = dict(file=path.name, sha256_packed=hashlib.sha256(packed).hexdigest(),
                  sha256_raw=hashlib.sha256(raw).hexdigest(), bond=row["bond"],
                  probabilities=probs, absolute_errors=errors, max_abs_error=max(errors),
                  p_detect_relative_error_fraction=rel, raw_sum=math.fsum(probs),
                  normalization_residual=residual, strict_four_state_passed=passed,
                  work_union_upper_bound=row["work_any_nonzero_union_upper_bound"],
                  work_status=row["work_accuracy_status"], execution=row["execution_status"])
    return row, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lower_bond", type=Path, help="new bond-64 per-setting JSON")
    p.add_argument("higher_bond", type=Path, help="archived single-thread bond-256 per-setting JSON or JSON.GZ")
    p.add_argument("--out", type=Path, required=True, help="new output filename")
    args = p.parse_args()
    a, low = read_probe(args.lower_bond)
    b, high = read_probe(args.higher_bond)
    classical_raw = (Path(__file__).resolve().parents[1] / REFERENCE).read_bytes()
    classical = json.loads(classical_raw)
    t15, = [row for row in classical["rows"] if row["steps"] == 15]
    if classical["order"] != STATUS or a["reference"] != t15["probabilities"]:
        raise ValueError("Independent archived same-grid reference differs.")
    if (a["bond"], b["bond"]) != (64, 256):
        raise ValueError("Expected bond caps of 64 and 256.")
    attrs = ("original_A_fingerprint", "compiled_A_fingerprint", "diagnostic_fingerprint", "reference",
             "truncation_threshold", "register_map", "qubits", "compiled_gates", "compiled_depth",
             "compiled_two_qubit_gates", "case", "blas_threads")
    for key in attrs:
        if a[key] != b[key]:
            raise ValueError(f"Cannot compare the runs with different {key}.")
    for key in ("config", "actual_profile", "compile", "measurement"):
        if a["identity"][key] != b["identity"][key]:
            raise ValueError(f"Identity mismatch for {key}.")
    opts_low = dict(a["backend_options"])
    opts_high = dict(b["backend_options"])
    for opts, cap in ((opts_low, 64), (opts_high, 256)):
        if opts.pop("matrix_product_state_max_bond_dimension") != cap:
            raise ValueError("Recorded backend cap does not match the case.")
    if opts_low != opts_high:
        raise ValueError("Backend options differ beyond the specified bond dimension.")
    result = dict(status_order=STATUS, tolerance=TOL, comparison="strict less than",
                  classical_reference_file=REFERENCE, classical_reference_sha256=hashlib.sha256(classical_raw).hexdigest(),
                  reference=a["reference"], original_A_fingerprint=a["original_A_fingerprint"],
                  compiled_A_fingerprint=a["compiled_A_fingerprint"],
                  diagnostic_fingerprint=a["diagnostic_fingerprint"],
                  compilation=a["identity"]["compile"], common_backend_options=opts_low,
                  model=a["case"], qubits=a["qubits"], compiled_gates=a["compiled_gates"],
                  compiled_two_qubit_gates=a["compiled_two_qubit_gates"], compiled_depth=a["compiled_depth"],
                  runs=[low, high])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for value in [low, high]:
        print(f"bond={value['bond']}: D={value['probabilities'][1]:.15g}, "
              f"max error={value['max_abs_error']:.9g}, normalization={value['normalization_residual']:.9g}, "
              f"strict={value['strict_four_state_passed']}")


if __name__ == "__main__":
    main()
