"""Postprocess detect-reach MPS outputs into rare-response estimates.

This script does not directly sample rare events.  It reads a standard
detect-reach MPS merged result and applies the target-table thinning factor:

    p_rare_Q = eta * p_detect_reach_Q
    SE_rare_Q = eta * SE_detect_reach_Q
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Dict[str, Any]:
    from result_storage import read_json
    return read_json(path)


def _read_target_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y", "ok")


def _sigma(x: float, x_se: float, ref: float, ref_se: float = 0.0) -> Optional[float]:
    den = math.sqrt(max(float(x_se), 0.0) ** 2 + max(float(ref_se), 0.0) ** 2)
    if den <= 0.0:
        return None
    return (float(x) - float(ref)) / den


def _load_reference(exact_json: str = "", mc_json: str = "") -> Dict[str, Any]:
    ref: Dict[str, Any] = {"kind": "", "p_detect": None, "se_detect": 0.0, "path": ""}
    if exact_json:
        js = _read_json(exact_json)
        final = js.get("final", js)
        ref.update(
            {
                "kind": "exact",
                "p_detect": _float_or_none(final.get("p_detect_reach", final.get("p_detect"))),
                "se_detect": 0.0,
                "path": exact_json,
            }
        )
        return ref
    if mc_json:
        js = _read_json(mc_json)
        ref.update(
            {
                "kind": "mc",
                "p_detect": _float_or_none(js.get("p_detect")),
                "se_detect": float(js.get("se_detect", 0.0) or 0.0),
                "path": mc_json,
            }
        )
    return ref


def postprocess(
    *,
    target_csv: str,
    mps_merged_json: str,
    exact_json: str = "",
    mc_json: str = "",
) -> Dict[str, Any]:
    targets = _read_target_csv(target_csv)
    mps = _read_json(mps_merged_json)
    ref = _load_reference(exact_json=exact_json, mc_json=mc_json)

    p_detect_q = _float_or_none(mps.get("p_detect", mps.get("p_detect_reach")))
    if p_detect_q is None:
        raise ValueError("MPS merged result does not contain p_detect")
    se_detect_q = float(mps.get("se_detect", mps.get("se_detect_reach", 0.0)) or 0.0)
    shots = int(mps.get("shots", 0) or 0)
    steps = int(mps.get("steps", targets[0].get("steps", 0) if targets else 0) or 0)

    z_detect = None
    if ref.get("p_detect") is not None:
        z_detect = _sigma(p_detect_q, se_detect_q, float(ref["p_detect"]), float(ref.get("se_detect", 0.0) or 0.0))

    rows: List[Dict[str, Any]] = []
    for target in targets:
        p_target = _float_or_none(target.get("p_target"))
        eta = _float_or_none(target.get("eta"))
        feasible = _boolish(target.get("feasible"))
        out = {
            "steps": steps,
            "shots": shots,
            "p_detect_reach_Q": p_detect_q,
            "SE_detect_reach_Q": se_detect_q,
            "Z_detect_reach": z_detect,
            "reference_kind": ref.get("kind", ""),
            "reference_path": ref.get("path", ""),
            "reference_p_detect": ref.get("p_detect", ""),
            "reference_se_detect": ref.get("se_detect", ""),
            "p_target": p_target,
            "eta": eta,
            "theta_eta": _float_or_none(target.get("theta_eta")),
            "feasible": feasible,
            "p_rare_Q": None,
            "SE_rare_Q": None,
            "p_rare_exact": _float_or_none(target.get("p_rare_exact")),
            "Z_rare": None,
            "relative_deviation_rare": None,
            "reason": target.get("reason", ""),
        }
        if feasible and eta is not None:
            p_rare_q = float(eta) * p_detect_q
            se_rare_q = abs(float(eta)) * se_detect_q
            p_rare_exact = _float_or_none(target.get("p_rare_exact"))
            out["p_rare_Q"] = p_rare_q
            out["SE_rare_Q"] = se_rare_q
            if p_rare_exact is not None:
                out["Z_rare"] = _sigma(p_rare_q, se_rare_q, p_rare_exact, 0.0)
                if p_rare_exact != 0.0:
                    out["relative_deviation_rare"] = (p_rare_q - p_rare_exact) / p_rare_exact
        rows.append(out)

    return {
        "phase": "rare_response_mps_postprocess",
        "target_csv": target_csv,
        "mps_merged_json": mps_merged_json,
        "steps": steps,
        "shots": shots,
        "p_detect_reach_Q": p_detect_q,
        "SE_detect_reach_Q": se_detect_q,
        "Z_detect_reach": z_detect,
        "reference": ref,
        "rows": rows,
        "notes": [
            "This is weighted rare-response postprocess from detect-reach MPS output.",
            "It is not direct rare-event sampling.",
        ],
    }


def _write_outputs(payload: Dict[str, Any], outdir: str, out_json: str = "", out_csv: str = "") -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = Path(out_json) if out_json else out_path / "rare_response_mps_postprocess.json"
    csv_path = Path(out_csv) if out_csv else out_path / "rare_response_mps_postprocess.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    fieldnames = [
        "steps",
        "shots",
        "p_detect_reach_Q",
        "SE_detect_reach_Q",
        "Z_detect_reach",
        "reference_kind",
        "reference_path",
        "reference_p_detect",
        "reference_se_detect",
        "p_target",
        "eta",
        "theta_eta",
        "feasible",
        "p_rare_Q",
        "SE_rare_Q",
        "p_rare_exact",
        "Z_rare",
        "relative_deviation_rare",
        "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in payload.get("rows", []):
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Postprocess detect-reach MPS output into rare-response estimates.")
    ap.add_argument("--target-csv", required=True, help="Exact rare-response target CSV.")
    ap.add_argument("--mps-merged-json", required=True, help="MPS merged/results_merged.json.")
    ap.add_argument("--exact-json", default="", help="Optional exact_reference JSON for detect Z-score.")
    ap.add_argument("--mc-json", default="", help="Optional MC results.json if exact JSON is not supplied.")
    ap.add_argument("--outdir", default=os.path.join("outputs", "extension", "rare_response", "mps_postprocess"))
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-csv", default="")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from output_guard import guard_output_arguments
    guard_output_arguments(args)
    payload = postprocess(
        target_csv=args.target_csv,
        mps_merged_json=args.mps_merged_json,
        exact_json=args.exact_json,
        mc_json=args.mc_json,
    )
    _write_outputs(payload, args.outdir, out_json=args.out_json, out_csv=args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
