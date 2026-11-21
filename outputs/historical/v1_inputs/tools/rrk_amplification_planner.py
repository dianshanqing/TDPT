"""Plan RRK Grover powers and query boundaries without running circuits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List


FRACTIONS = (0.0, 0.1, 0.3, 0.6, 0.9)


def _float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def ae_success_probability(p: float, m: int) -> float:
    p = min(1.0, max(0.0, float(p)))
    theta = math.asin(math.sqrt(p))
    return math.sin((2 * int(m) + 1) * theta) ** 2


def naive_histories(p: float, relative_error: float) -> float:
    p = float(p)
    r = float(relative_error)
    if p <= 0.0 or r <= 0.0:
        return math.inf
    return (1.0 - p) / (p * r * r)


def plan_target(p_target: float, fractions: List[float]) -> Dict[str, Any]:
    p = float(p_target)
    if not (0.0 < p <= 1.0):
        raise ValueError("target probabilities must be in (0, 1]")
    m_star = max(0.0, math.pi / (4.0 * math.sqrt(p)) - 0.5)
    oracle_calls_star = 2.0 * m_star + 1.0
    suggestions: List[Dict[str, Any]] = []
    seen = set()
    for frac in fractions:
        raw = float(frac) * m_star
        power = max(0, int(round(raw)))
        if power in seen:
            continue
        seen.add(power)
        suggestions.append(
            {
                "fraction_of_m_star": float(frac),
                "suggested_power_float": raw,
                "suggested_power": int(power),
                "oracle_calls_suggested": int(2 * power + 1),
                "expected_success_probability": float(ae_success_probability(p, power)),
            }
        )
    req = {
        "naive_MC_required_histories_R=0.5": naive_histories(p, 0.5),
        "naive_MC_required_histories_R=0.2": naive_histories(p, 0.2),
        "naive_MC_required_histories_R=0.1": naive_histories(p, 0.1),
    }
    return {
        "p_target": p,
        "m_star": float(m_star),
        "oracle_calls_star": float(oracle_calls_star),
        **req,
        "suggestions": suggestions,
    }


def build_rows(plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for plan in plans:
        for suggestion in plan["suggestions"]:
            rows.append(
                {
                    "p_target": plan["p_target"],
                    "m_star": plan["m_star"],
                    "oracle_calls_star": plan["oracle_calls_star"],
                    "fraction_of_m_star": suggestion["fraction_of_m_star"],
                    "suggested_power_float": suggestion["suggested_power_float"],
                    "suggested_power": suggestion["suggested_power"],
                    "oracle_calls_suggested": suggestion["oracle_calls_suggested"],
                    "expected_success_probability": suggestion["expected_success_probability"],
                    "naive_MC_required_histories_R=0.5": plan["naive_MC_required_histories_R=0.5"],
                    "naive_MC_required_histories_R=0.2": plan["naive_MC_required_histories_R=0.2"],
                    "naive_MC_required_histories_R=0.1": plan["naive_MC_required_histories_R=0.1"],
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "p_target",
        "m_star",
        "oracle_calls_star",
        "fraction_of_m_star",
        "suggested_power_float",
        "suggested_power",
        "oracle_calls_suggested",
        "expected_success_probability",
        "naive_MC_required_histories_R=0.5",
        "naive_MC_required_histories_R=0.2",
        "naive_MC_required_histories_R=0.1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate RRK amplification boundary plans.")
    ap.add_argument("--targets", default="1e-6,1e-7,1e-8,1e-9,1e-10")
    ap.add_argument("--fractions", default="0,0.1,0.3,0.6,0.9")
    ap.add_argument("--outdir", default=os.path.join("outputs", "extension", "rrk", "plan"))
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-json", default="")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    targets = _float_list(args.targets)
    fractions = _float_list(args.fractions)
    plans = [plan_target(p, fractions) for p in targets]
    rows = build_rows(plans)

    outdir = Path(args.outdir)
    out_csv = Path(args.out_csv) if args.out_csv else outdir / "rrk_amplification_plan.csv"
    out_json = Path(args.out_json) if args.out_json else outdir / "rrk_amplification_plan.json"
    _write_csv(out_csv, rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "rrk_amplification_planner",
                "targets": targets,
                "fractions": fractions,
                "plans": plans,
                "notes": [
                    "No quantum circuits are built or run by this planner.",
                    "m_star is the approximate first Grover maximum for a Bernoulli target probability.",
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
