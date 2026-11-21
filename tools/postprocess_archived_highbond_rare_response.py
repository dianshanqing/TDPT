"""Batch rare-response postprocess for archived high-bond MPS outputs.

This script only reads existing merged/results_merged.json files.  It does not
launch new MPS simulations.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
for p in (TOOLS, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postprocess_rare_response_mps import postprocess
from result_storage import logical_path, read_json, result_exists, result_files


PATH_RE = re.compile(
    r"test_s(?P<steps>\d+)_shards_total(?P<shots>\d+)_.*?_chunk(?P<chunk>\d+)_(?P<bond>\d+)_(?P<trunc>[^\\/]+)$"
)


def _parse_meta(run_dir: Path, merged_json: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    m = PATH_RE.search(run_dir.name)
    meta = {
        "steps": int(payload.get("steps", 0) or 0),
        "shots": int(payload.get("shots", 0) or 0),
        "bond": "",
        "truncation": "",
        "shot_chunk": "",
        "run_dir": str(run_dir),
        "mps_merged_json": str(merged_json),
    }
    if m:
        meta.update(
            {
                "steps": int(m.group("steps")),
                "shots": int(m.group("shots")),
                "bond": int(m.group("bond")),
                "truncation": m.group("trunc"),
                "shot_chunk": int(m.group("chunk")),
            }
        )
    return meta


def _read_json(path: Path) -> Dict[str, Any]:
    return read_json(path)


def _discover(root: Path, steps_filter: Optional[set[int]] = None) -> List[Path]:
    pattern = "test_s*_shards_total*_sum*_seed*_chunk*_*_*/merged/results_merged.json"
    paths = [logical_path(path) for path in result_files(root, pattern)]
    if steps_filter:
        out = []
        for p in paths:
            m = PATH_RE.search(p.parents[1].name)
            if m and int(m.group("steps")) in steps_filter:
                out.append(p)
        paths = out
    return sorted(paths, key=lambda p: p.parents[1].name)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "steps",
        "bond",
        "shots",
        "truncation",
        "shot_chunk",
        "p_detect_reach_Q",
        "SE_detect_reach_Q",
        "exact_p_detect_reach",
        "Z_detect_reach",
        "p_target",
        "eta",
        "theta_eta",
        "p_rare_Q",
        "SE_rare_Q",
        "p_rare_exact",
        "Z_rare",
        "relative_deviation_rare",
        "feasible",
        "run_dir",
        "mps_merged_json",
        "target_csv",
        "exact_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch postprocess archived high-bond MPS outputs into rare-response rows.")
    ap.add_argument("--root", default=os.path.join("outputs", "test_s12_s15_bond_trunc"))
    ap.add_argument("--steps", default="12,15", help="Comma-separated steps to include.")
    ap.add_argument("--exact-root", default=os.path.join("outputs", "extension", "rare_response", "exact_reference"))
    ap.add_argument("--rare-root", default=os.path.join("outputs", "extension", "rare_response", "targets"))
    ap.add_argument("--outdir", default=os.path.join("outputs", "extension", "rare_response", "archived_validation"))
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-json", default="")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from output_guard import guard_output_arguments
    guard_output_arguments(args)
    steps_filter = {int(x.strip()) for x in str(args.steps).split(",") if x.strip()}
    root = Path(args.root)
    outdir = Path(args.outdir)
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for merged_json in _discover(root, steps_filter):
        run_dir = merged_json.parents[1]
        mps_payload = _read_json(merged_json)
        meta = _parse_meta(run_dir, merged_json, mps_payload)
        steps = int(meta["steps"])
        exact_json = Path(args.exact_root) / f"exact_s{steps}.json"
        target_csv = Path(args.rare_root) / f"rare_response_targets_s{steps}.csv"
        if not target_csv.exists() or not result_exists(exact_json):
            skipped.append(
                {
                    "mps_merged_json": str(merged_json),
                    "reason": "missing exact_json or rare target csv",
                    "exact_json": str(exact_json),
                    "target_csv": str(target_csv),
                }
            )
            continue

        payload = postprocess(
            target_csv=str(target_csv),
            mps_merged_json=str(merged_json),
            exact_json=str(exact_json),
        )
        for row in payload.get("rows", []):
            out = dict(row)
            out.update(meta)
            out["exact_p_detect_reach"] = row.get("reference_p_detect", "")
            out["target_csv"] = str(target_csv)
            out["exact_json"] = str(exact_json)
            rows.append(out)

    rows = sorted(rows, key=lambda r: (int(r.get("steps", 0)), int(r.get("bond", 0)), int(r.get("shots", 0)), float(r.get("p_target", 0.0) or 0.0)))
    out_csv = Path(args.out_csv) if args.out_csv else outdir / "archived_highbond_rare_response.csv"
    out_json = Path(args.out_json) if args.out_json else outdir / "archived_highbond_rare_response.json"
    _write_csv(out_csv, rows)
    _write_json(
        out_json,
        {
            "phase": "archived_highbond_rare_response",
            "root": str(root),
            "rows": rows,
            "skipped": skipped,
            "notes": [
                "Existing archived MPS merged outputs were postprocessed only.",
                "No new high-step MPS simulation was launched.",
            ],
        },
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    if skipped:
        print(f"Skipped {len(skipped)} inputs with missing references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
