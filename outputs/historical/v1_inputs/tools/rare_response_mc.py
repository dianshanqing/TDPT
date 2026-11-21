"""Weighted and analog MC reference for rare detect response.

The transport trajectory is unchanged.  A rare response is tallied only after
the original detect terminal event:
  analog  : draw a Bernoulli(eta) response coin;
  weighted: add deterministic weight eta to each detect reach.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for p in (ROOT, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import default_config
from exact_reference_solver import propagate_exact
from mc import run_mc_rare_response


def _sci_label(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return value


def _default_outdir(args: argparse.Namespace, eta: Optional[float]) -> Path:
    if args.outdir:
        return Path(args.outdir)
    if eta is not None:
        label = f"eta{_sci_label(float(eta))}"
    elif args.p_target is not None:
        label = f"ptarget{_sci_label(float(args.p_target))}"
    else:
        label = "eta_unknown"
    return Path("outputs") / "rare_response_mc" / (
        f"mcrare_s{int(args.steps)}_h{int(args.histories)}_{label}_seed{int(args.seed)}"
    )


def _write_payload(outdir: Path, payload: Dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(_json_clean(payload), f, indent=2, ensure_ascii=False, allow_nan=False)

    flat = dict(payload)
    naive = flat.pop("naive_MC_required_histories", {})
    if isinstance(naive, dict):
        for k, v in naive.items():
            flat[f"naive_MC_required_histories_{k}"] = v

    fieldnames = [
        "phase",
        "steps",
        "histories",
        "seed",
        "eta",
        "p_target",
        "p_detect_reach_for_eta",
        "feasible",
        "p_detect_reach_hat",
        "p_rare_analog_hat",
        "p_rare_weighted_hat",
        "se_detect_reach",
        "se_rare_analog",
        "se_rare_weighted",
        "detect_reach_count",
        "analog_rare_count",
        "naive_MC_required_histories_R=0.5",
        "naive_MC_required_histories_R=0.2",
        "naive_MC_required_histories_R=0.1",
        "reason",
    ]
    with (outdir / "rare_mc_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({k: _json_clean(flat.get(k, "")) for k in fieldnames})


def _infer_eta(args: argparse.Namespace, cfg) -> Dict[str, Any]:
    if args.eta is not None:
        eta = float(args.eta)
        return {
            "eta": eta,
            "p_detect_reach_for_eta": None,
            "feasible": 0.0 <= eta <= 1.0,
            "reason": "eta_supplied" if 0.0 <= eta <= 1.0 else "eta_out_of_range",
        }

    if args.p_target is None:
        raise ValueError("provide --eta, or provide --p-target to infer eta from exact detect reach")

    p_target = float(args.p_target)
    p_detect = args.p_detect_reach
    if p_detect is None:
        rows = propagate_exact(
            cfg,
            int(args.steps),
            unitary_matched_energy=bool(args.unitary_matched_energy),
        )
        p_detect = float(rows[-1]["p_detect_reach"])
    else:
        p_detect = float(p_detect)

    if p_detect <= 0.0:
        return {
            "eta": None,
            "p_detect_reach_for_eta": p_detect,
            "feasible": False,
            "reason": "no_detect_reach",
        }

    eta = p_target / p_detect
    if eta > 1.0:
        return {
            "eta": eta,
            "p_detect_reach_for_eta": p_detect,
            "feasible": False,
            "reason": "eta_gt_1",
        }
    return {
        "eta": eta,
        "p_detect_reach_for_eta": p_detect,
        "feasible": True,
        "reason": "ok",
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Rare-response analog/weighted MC reference.")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--histories", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eta", type=float, default=None, help="Rare response efficiency in [0,1].")
    ap.add_argument("--p-target", type=float, default=None, help="Infer eta = p_target / p_detect_reach.")
    ap.add_argument("--p-detect-reach", type=float, default=None, help="Optional detect reach probability for eta inference.")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--unitary-matched-energy", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    cfg = default_config()
    inferred = _infer_eta(args, cfg)
    eta = inferred.get("eta")
    outdir = _default_outdir(args, eta if eta is not None else None)

    base_payload: Dict[str, Any] = {
        "phase": "rare_response_mc",
        "steps": int(args.steps),
        "histories": int(args.histories),
        "seed": int(args.seed),
        "eta": eta,
        "p_target": None if args.p_target is None else float(args.p_target),
        "p_detect_reach_for_eta": inferred.get("p_detect_reach_for_eta"),
        "feasible": bool(inferred.get("feasible")),
        "reason": str(inferred.get("reason", "")),
        "unitary_matched_energy": bool(args.unitary_matched_energy),
    }

    if not bool(inferred.get("feasible")):
        _write_payload(outdir, base_payload)
        print(f"Wrote infeasible rare-response MC record to {outdir}")
        return 2

    res = run_mc_rare_response(
        cfg,
        n_steps=int(args.steps),
        histories=int(args.histories),
        eta=float(eta),
        seed=int(args.seed),
        unitary_matched_energy=bool(args.unitary_matched_energy),
    )
    payload = base_payload
    payload.update(asdict(res))
    _write_payload(outdir, payload)
    print(f"Wrote rare-response MC outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
