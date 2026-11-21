"""Exact dynamic-programming reference for the VRQO transport benchmark.

This script propagates the finite Markov state distribution without random
sampling.  The active state includes position, energy group, and direction;
terminal probability is accumulated in the same order as mc.run_mc:
boundary check, detector ROI check, material interaction, absorption,
scatter/energy update, and controlled move.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TransportConfig, default_config, validate_config
from geometry import clamp_pos, is_boundary, is_in_roi, material_at


DIRS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

STATUS_CODES = {
    "alive": "00",
    "detect": "01",
    "absorb": "10",
    "boundary": "11",
}

DEFAULT_TARGETS = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)


def _move(cfg: TransportConfig, x: int, y: int, z: int, direction: int) -> Tuple[int, int, int]:
    dx, dy, dz = DIRS[int(direction)]
    return clamp_pos(cfg, x + dx, y + dy, z + dz)


def _add_active(
    active: Dict[Tuple[int, int, int, int, int], float],
    cfg: TransportConfig,
    x: int,
    y: int,
    z: int,
    g: int,
    direction: int,
    prob: float,
) -> None:
    if prob <= 0.0:
        return
    nx, ny, nz = _move(cfg, x, y, z, direction)
    active[(nx, ny, nz, int(g), int(direction))] += float(prob)


def _energy_branches(
    cfg: TransportConfig,
    g: int,
    q_down: float,
    *,
    unitary_matched_energy: bool,
) -> Iterable[Tuple[int, float]]:
    groups = int(cfg.groups.n_groups)
    q_down = float(q_down)
    if g < groups - 1 and q_down > 0.0:
        if q_down < 1.0:
            yield int(g), 1.0 - q_down
        if unitary_matched_energy:
            yield int((g + 1) % groups), q_down
        else:
            yield int(min(g + 1, groups - 1)), q_down
    else:
        yield int(g), 1.0


def propagate_exact(
    cfg: TransportConfig,
    n_steps: int,
    *,
    unitary_matched_energy: bool = False,
) -> List[Dict[str, Any]]:
    validate_config(cfg)
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1")

    src = cfg.source
    active: Dict[Tuple[int, int, int, int, int], float] = {
        (int(src.x0), int(src.y0), int(src.z0), int(src.g0), int(getattr(src, "dir0", 0))): 1.0
    }
    terminal = {"detect": 0.0, "absorb": 0.0, "boundary": 0.0}
    rows: List[Dict[str, Any]] = []

    for step in range(1, int(n_steps) + 1):
        next_active: Dict[Tuple[int, int, int, int, int], float] = defaultdict(float)
        terminal_this = {"detect": 0.0, "absorb": 0.0, "boundary": 0.0}

        for (x, y, z, g, direction), prob in active.items():
            prob = float(prob)
            if prob <= 0.0:
                continue

            if is_boundary(cfg, x, y, z):
                terminal["boundary"] += prob
                terminal_this["boundary"] += prob
                continue

            if is_in_roi(cfg, x, y, z):
                terminal["detect"] += prob
                terminal_this["detect"] += prob
                continue

            mat_name = material_at(cfg, x, y, z)
            mat = getattr(cfg.mats, mat_name)
            p_int = float(mat.p_int[g])
            p_abs = float(mat.p_abs[g])
            q_down = float(mat.q_down[g])

            _add_active(next_active, cfg, x, y, z, g, direction, prob * (1.0 - p_int))

            interact = prob * p_int
            if interact <= 0.0:
                continue

            absorbed = interact * p_abs
            if absorbed > 0.0:
                terminal["absorb"] += absorbed
                terminal_this["absorb"] += absorbed

            scatter_mass = interact * (1.0 - p_abs)
            if scatter_mass <= 0.0:
                continue

            for new_direction in range(6):
                dir_mass = scatter_mass / 6.0
                for new_g, g_prob in _energy_branches(
                    cfg,
                    g,
                    q_down,
                    unitary_matched_energy=unitary_matched_energy,
                ):
                    _add_active(
                        next_active,
                        cfg,
                        x,
                        y,
                        z,
                        new_g,
                        new_direction,
                        dir_mass * float(g_prob),
                    )

        active = dict(next_active)
        p_survive = float(sum(active.values()))
        sum_prob = float(terminal["detect"] + terminal["absorb"] + terminal["boundary"] + p_survive)
        rows.append(
            {
                "step": int(step),
                "p_detect_reach": float(terminal["detect"]),
                "p_absorb": float(terminal["absorb"]),
                "p_boundary": float(terminal["boundary"]),
                "p_survive": p_survive,
                "sum_prob": sum_prob,
                "active_states": int(len(active)),
                "terminal_this_step": {k: float(v) for k, v in terminal_this.items()},
            }
        )

    return rows


def compute_rare_targets(p_detect_reach: float, targets: Iterable[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p_detect_reach = float(p_detect_reach)
    for target in targets:
        p_target = float(target)
        row: Dict[str, Any] = {
            "p_detect_reach": p_detect_reach,
            "p_target": p_target,
            "eta": None,
            "theta_eta": None,
            "p_rare_exact": 0.0,
            "feasible": False,
            "reason": "",
        }
        if p_detect_reach <= 0.0:
            row["reason"] = "no_detect_reach"
        else:
            eta = p_target / p_detect_reach
            row["eta"] = float(eta)
            if eta <= 1.0:
                theta = 2.0 * math.asin(math.sqrt(max(eta, 0.0)))
                row.update(
                    {
                        "theta_eta": float(theta),
                        "p_rare_exact": float(p_detect_reach * eta),
                        "feasible": True,
                        "reason": "ok",
                    }
                )
            else:
                row.update(
                    {
                        "p_rare_exact": float(p_target),
                        "feasible": False,
                        "reason": "eta_gt_1",
                    }
                )
        rows.append(row)
    return rows


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_steps(args: argparse.Namespace) -> List[int]:
    steps: List[int] = []
    if args.steps is not None:
        steps.append(int(args.steps))
    if args.steps_list:
        steps.extend(int(x.strip()) for x in str(args.steps_list).split(",") if x.strip())
    steps = sorted(set(steps))
    if not steps:
        raise ValueError("provide --steps or --steps-list")
    if min(steps) < 1:
        raise ValueError("all steps must be >= 1")
    return steps


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_mc_reference(step: int, mc_root: Path) -> Optional[Path]:
    patterns = [
        str(mc_root / f"mc_s{int(step)}_h*_seed*" / "results.json"),
        str(mc_root / f"*s{int(step)}*" / "results.json"),
    ]
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    matches = sorted(set(matches))
    return Path(matches[0]) if matches else None


def _compare_mc(final_row: Dict[str, Any], mc_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, exact_key in (
        ("detect", "p_detect_reach"),
        ("absorb", "p_absorb"),
        ("boundary", "p_boundary"),
        ("survive", "p_survive"),
    ):
        exact_p = float(final_row[exact_key])
        mc_p = float(mc_json.get(f"p_{name}", 0.0))
        mc_se = float(mc_json.get(f"se_{name}", 0.0))
        diff = exact_p - mc_p
        z = None if mc_se <= 0.0 else diff / mc_se
        within = (abs(diff) <= 1e-12) if z is None else (abs(float(z)) <= 3.0)
        out.append(
            {
                "state": name,
                "exact_p": exact_p,
                "mc_p": mc_p,
                "mc_se": mc_se,
                "diff": float(diff),
                "z_vs_mc": None if z is None else float(z),
                "within_3sigma": bool(within),
            }
        )
    return out


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Exact DP/Markov reference solver for VRQO.")
    ap.add_argument("--steps", type=int, default=None, help="Single step count.")
    ap.add_argument("--steps-list", default="", help="Comma-separated step counts, e.g. 7,10,12,15.")
    ap.add_argument("--seed", type=int, default=1234, help="Accepted for CLI parity; exact solver does not use it.")
    ap.add_argument("--outdir", default=os.path.join("outputs", "extension", "rare_response", "exact_reference"))
    ap.add_argument("--rare-outdir", default=os.path.join("outputs", "extension", "rare_response", "targets"))
    ap.add_argument("--targets", default=",".join(f"{x:g}" for x in DEFAULT_TARGETS))
    ap.add_argument("--mc-root", default=os.path.join("outputs", "MC"))
    ap.add_argument("--unitary-matched-energy", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    steps = _parse_steps(args)
    targets = _parse_float_list(args.targets)
    cfg = default_config()

    all_rows = propagate_exact(
        cfg,
        max(steps),
        unitary_matched_energy=bool(args.unitary_matched_energy),
    )

    outdir = Path(args.outdir)
    rare_outdir = Path(args.rare_outdir)
    mc_root = Path(args.mc_root)
    summary_rows: List[Dict[str, Any]] = []

    for step in steps:
        step_rows = all_rows[: int(step)]
        final = dict(step_rows[-1])
        mc_path = _find_mc_reference(step, mc_root)
        mc_compare: List[Dict[str, Any]] = []
        mc_ref: Optional[Dict[str, Any]] = None
        if mc_path is not None:
            mc_ref = {"path": str(mc_path)}
            mc_compare = _compare_mc(final, _read_json(str(mc_path)))
            mc_ref["comparison"] = mc_compare

        payload = {
            "phase": "exact_reference",
            "steps": int(step),
            "seed_ignored": int(args.seed),
            "unitary_matched_energy": bool(args.unitary_matched_energy),
            "status_codes": STATUS_CODES,
            "step_summaries": step_rows,
            "final": final,
            "mc_reference": mc_ref,
            "notes": [
                "The exact solver is deterministic; --seed is accepted only for output metadata.",
                "p_survive is the active alive mass after the requested number of steps.",
            ],
        }
        _dump_json(outdir / f"exact_s{int(step)}.json", payload)

        rare_rows = compute_rare_targets(float(final["p_detect_reach"]), targets)
        for row in rare_rows:
            row["steps"] = int(step)
        _write_csv(
            rare_outdir / f"rare_response_targets_s{int(step)}.csv",
            rare_rows,
            [
                "steps",
                "p_detect_reach",
                "p_target",
                "eta",
                "theta_eta",
                "p_rare_exact",
                "feasible",
                "reason",
            ],
        )

        max_abs_z = None
        if mc_compare:
            z_vals = [abs(float(r["z_vs_mc"])) for r in mc_compare if r.get("z_vs_mc") is not None]
            max_abs_z = max(z_vals) if z_vals else None
        summary_rows.append(
            {
                "steps": int(step),
                "p_detect_reach": final["p_detect_reach"],
                "p_absorb": final["p_absorb"],
                "p_boundary": final["p_boundary"],
                "p_survive": final["p_survive"],
                "sum_prob": final["sum_prob"],
                "active_states": final["active_states"],
                "mc_reference": "" if mc_path is None else str(mc_path),
                "mc_max_abs_z": "" if max_abs_z is None else max_abs_z,
            }
        )

    _write_csv(
        outdir / "exact_summary.csv",
        summary_rows,
        [
            "steps",
            "p_detect_reach",
            "p_absorb",
            "p_boundary",
            "p_survive",
            "sum_prob",
            "active_states",
            "mc_reference",
            "mc_max_abs_z",
        ],
    )

    print(f"Wrote exact outputs to {outdir}")
    print(f"Wrote rare-response target tables to {rare_outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

