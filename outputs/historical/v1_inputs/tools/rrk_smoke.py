"""RRK exact/MC/quantum smoke workflow."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rrk import exact_probability, infer_eta, run_rrk_analytic_fixed_depth, run_rrk_fixed_depth, run_rrk_mc


def _float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "p_target",
        "p_reach_exact",
        "eta",
        "feasible",
        "p_rare_exact",
        "theta_eta",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _json_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _write_quantum_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sampler_type",
        "backend_method",
        "m",
        "oracle_calls",
        "shots",
        "success_count",
        "failure_count",
        "success_probability",
        "expected_success_probability",
        "success_se",
        "circuit_depth",
        "gate_count",
        "qubit_count",
        "elapsed_time_sec",
        "build_time_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            out["gate_count"] = _json_cell(row.get("gate_count", ""))
            w.writerow(out)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a low-cost reduced response kernel smoke workflow.")
    ap.add_argument("--layers", default="0.1,0.1,0.1", help="Comma-separated layer transmission probabilities.")
    ap.add_argument("--targets", default="1e-6,1e-7,1e-8,1e-9,1e-10")
    ap.add_argument("--p-target", type=float, default=1e-6, help="Target used for MC and quantum smoke.")
    ap.add_argument("--eta", type=float, default=-1.0, help="Override eta for MC/quantum smoke.")
    ap.add_argument("--histories", type=int, default=5000)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--powers", default="0,1,2")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-quantum", action="store_true")
    ap.add_argument("--analytic-sampler", action="store_true", help="Use analytic binomial RRK fixed-depth sampling instead of circuits.")
    ap.add_argument("--rrk-backend", choices=["statevector", "automatic", "mps"], default="statevector")
    ap.add_argument("--rrk-max-parallel-threads", type=int, default=0, help="Aer max_parallel_threads; 0 lets Aer choose.")
    ap.add_argument("--outdir", default=os.path.join("outputs", "extension", "scratch", "rrk_smoke"))
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    layers = _float_list(args.layers)
    targets = _float_list(args.targets)
    powers = _int_list(args.powers)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target_rows: List[Dict[str, Any]] = []
    for target in targets:
        row = infer_eta(layers, target)
        eta = row.get("eta")
        p_rare = ""
        theta = ""
        if row.get("feasible") and eta is not None:
            ex = exact_probability(layers, float(eta))
            p_rare = ex.p_rare_exact
            import math

            theta = 2.0 * math.asin(math.sqrt(float(eta)))
        target_rows.append({**row, "p_rare_exact": p_rare, "theta_eta": theta})
    _write_csv(outdir / "rrk_target_table.csv", target_rows)

    eta = float(args.eta)
    if eta < 0.0:
        inferred = infer_eta(layers, float(args.p_target))
        if not inferred.get("feasible"):
            raise SystemExit(f"Requested p_target is infeasible for RRK layers: {inferred}")
        eta = float(inferred["eta"])

    exact = exact_probability(layers, eta)
    mc = run_rrk_mc(layers, eta, histories=int(args.histories), seed=int(args.seed))
    quantum = None
    if not args.skip_quantum:
        if args.analytic_sampler:
            quantum = run_rrk_analytic_fixed_depth(
                layers,
                eta,
                powers=powers,
                shots=int(args.shots),
                seed=int(args.seed),
            )
        else:
            backend_options: Dict[str, Any] = {}
            if int(args.rrk_max_parallel_threads) > 0:
                backend_options["max_parallel_threads"] = int(args.rrk_max_parallel_threads)
            quantum = run_rrk_fixed_depth(
                layers,
                eta,
                powers=powers,
                shots=int(args.shots),
                seed=int(args.seed),
                backend_method=str(args.rrk_backend),
                backend_options=backend_options,
            )

    payload = {
        "phase": "rrk_smoke",
        "layers": layers,
        "targets": targets,
        "selected_p_target": float(args.p_target),
        "selected_eta": eta,
        "sampler_type": "" if quantum is None else quantum.get("sampler_type", ""),
        "backend_method": "" if quantum is None else quantum.get("backend_method", ""),
        "exact": exact.__dict__,
        "mc": mc,
        "quantum": quantum,
        "notes": [
            "RRK is a reduced layered attenuation response kernel.",
            "It is intended for low-cost quantum AE validation, not as a full 3D transport replacement.",
            "Analytic sampler rows are for MLE/power cross-checks and are not quantum-circuit evidence.",
        ],
    }
    with (outdir / "rrk_smoke_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    if quantum is not None:
        _write_quantum_rows_csv(outdir / "rrk_quantum_rows.csv", quantum.get("rows", []))

    print(f"Wrote {outdir / 'rrk_target_table.csv'}")
    print(f"Wrote {outdir / 'rrk_smoke_results.json'}")
    if quantum is not None:
        print(f"Wrote {outdir / 'rrk_quantum_rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
