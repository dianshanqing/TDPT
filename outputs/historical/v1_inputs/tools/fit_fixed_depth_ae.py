"""Maximum-likelihood fit for fixed-depth AE / MLAE diagnostic rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ae_power_probability(p: float, m: int) -> float:
    p = min(1.0, max(0.0, float(p)))
    theta = math.asin(math.sqrt(p))
    return math.sin((2 * int(m) + 1) * theta) ** 2


def _term_nll(prob: float, success: int, shots: int) -> float:
    fail = int(shots) - int(success)
    if prob <= 0.0:
        return 0.0 if success == 0 else math.inf
    if prob >= 1.0:
        return 0.0 if fail == 0 else math.inf
    return -(success * math.log(prob) + fail * math.log(1.0 - prob))


def negative_log_likelihood(p: float, rows: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        m = int(row["m"])
        shots = int(row["shots"])
        success = int(row["success_count"])
        total += _term_nll(ae_power_probability(p, m), success, shots)
    return total


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            js = json.load(f)
        rows = js.get("rows", [])
        return [dict(r) for r in rows]
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _to_fit_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for row in rows:
        shots = int(float(row.get("shots", 0) or 0))
        success = int(float(row.get("success_count", 0) or 0))
        m = int(float(row.get("m", 0) or 0))
        if shots <= 0:
            continue
        out.append({"m": m, "shots": shots, "success_count": success})
    if not out:
        raise ValueError("no valid fixed-depth AE rows found")
    return out


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _detect_local_minima(profile: List[Tuple[float, float]], tolerance: Optional[float] = None, max_count: int = 25) -> List[Dict[str, float]]:
    mins: List[Dict[str, float]] = []
    if len(profile) < 3:
        return mins
    finite_nll = [nll for _, nll in profile if _finite(nll)]
    if not finite_nll:
        return mins
    best = min(finite_nll)
    for i in range(1, len(profile) - 1):
        p, nll = profile[i]
        if not _finite(nll):
            continue
        delta = nll - best
        if tolerance is not None and delta > tolerance:
            continue
        if nll <= profile[i - 1][1] and nll <= profile[i + 1][1]:
            mins.append({"p": float(p), "negative_log_likelihood": float(nll), "delta_nll": float(nll - best)})
    # Boundary minima matter for all-zero/all-one data.
    for i in (0, len(profile) - 1):
        p, nll = profile[i]
        if not _finite(nll):
            continue
        neighbor = profile[1][1] if i == 0 else profile[-2][1]
        delta = nll - best
        if tolerance is not None and delta > tolerance:
            continue
        if nll <= neighbor:
            mins.append({"p": float(p), "negative_log_likelihood": float(nll), "delta_nll": float(nll - best)})
    return sorted(mins, key=lambda x: (x["delta_nll"], x["p"]))[:max_count]


def _warnings(
    rows: List[Dict[str, int]],
    p_mle: float,
    local_minima: List[Dict[str, float]],
    alias_candidates: Optional[List[Dict[str, float]]] = None,
    p_min: float = 0.0,
    p_max: float = 1.0,
    log_grid: bool = False,
    theta_grid: bool = False,
    expected_low_p: bool = False,
) -> List[str]:
    warnings: List[str] = []
    successes = sum(int(r["success_count"]) for r in rows)
    shots = sum(int(r["shots"]) for r in rows)
    if successes == 0:
        warnings.append("zero_success_all_powers")
    if successes == shots:
        warnings.append("all_success_all_powers")
    if p_mle <= 1e-15:
        warnings.append("boundary_solution_p0")
    if p_mle >= 1.0 - 1e-15:
        warnings.append("boundary_solution_p1")
    span = max(float(p_max) - float(p_min), 0.0)
    edge_tol = max(span * 1e-10, 1e-18)
    if float(p_min) > 0.0 and p_mle <= float(p_min) + edge_tol:
        warnings.append("boundary_solution_p_min")
    if p_mle >= float(p_max) - edge_tol:
        warnings.append("boundary_solution_p_max")
    if len(local_minima) > 1 or (alias_candidates is not None and len(alias_candidates) > 1):
        warnings.append("multi_peak_likelihood")
    if expected_low_p and not (log_grid or theta_grid):
        warnings.append("expected_low_p_linear_grid_risk")
    if expected_low_p and float(p_max) > 1e-3:
        warnings.append("expected_low_p_wide_range_alias_risk")
    return warnings


def _unique_sorted(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in sorted(float(v) for v in values if not math.isnan(float(v))):
        if not out or value != out[-1]:
            out.append(value)
    return out


def _make_grid(
    p_min: float,
    p_max: float,
    grid_size: int,
    log_grid: bool,
    theta_grid: bool,
    exact_p: Optional[float],
) -> List[float]:
    if theta_grid and log_grid:
        raise ValueError("--log-grid and --theta-grid are mutually exclusive")
    p_min = max(0.0, float(p_min))
    p_max = min(1.0, float(p_max))
    if p_max < p_min:
        raise ValueError("--p-max must be >= --p-min")
    grid_size = max(101, int(grid_size))
    values: List[float] = [p_min, p_max]
    if p_min <= 0.0 <= p_max:
        values.append(0.0)
    if exact_p is not None and p_min <= float(exact_p) <= p_max:
        values.append(float(exact_p))
    if p_max == p_min:
        return _unique_sorted(values)
    if theta_grid:
        t0 = math.asin(math.sqrt(p_min))
        t1 = math.asin(math.sqrt(p_max))
        for i in range(grid_size):
            t = t0 + (t1 - t0) * i / float(grid_size - 1)
            values.append(math.sin(t) ** 2)
    elif log_grid:
        if p_max <= 0.0:
            return _unique_sorted(values)
        positive_min = p_min if p_min > 0.0 else max(p_max * 1e-12, 1e-300)
        positive_min = min(positive_min, p_max)
        lo = math.log(positive_min)
        hi = math.log(p_max)
        for i in range(grid_size):
            x = lo + (hi - lo) * i / float(grid_size - 1)
            values.append(math.exp(x))
    else:
        for i in range(grid_size):
            values.append(p_min + (p_max - p_min) * i / float(grid_size - 1))
    return _unique_sorted(values)


def _axis_value(p: float, log_grid: bool, theta_grid: bool, positive_floor: float) -> float:
    if theta_grid:
        return math.asin(math.sqrt(min(1.0, max(0.0, p))))
    if log_grid:
        return math.log(max(float(p), positive_floor))
    return float(p)


def _axis_to_p(x: float, log_grid: bool, theta_grid: bool) -> float:
    if theta_grid:
        return math.sin(x) ** 2
    if log_grid:
        return math.exp(x)
    return x


def _refine_best(
    rows: List[Dict[str, int]],
    profile: List[Tuple[float, float]],
    best_index: int,
    best_p: float,
    best_nll: float,
    p_min: float,
    p_max: float,
    log_grid: bool,
    theta_grid: bool,
) -> Tuple[float, float]:
    left_p = profile[best_index - 1][0] if best_index > 0 else p_min
    right_p = profile[best_index + 1][0] if best_index + 1 < len(profile) else p_max
    if log_grid and best_p <= 0.0:
        log_grid = False
    positive_floor = max(min(x for x in (left_p, best_p, right_p, p_max) if x > 0.0), 1e-300) if any(x > 0.0 for x in (left_p, best_p, right_p, p_max)) else 1e-300
    lo_x = _axis_value(left_p, log_grid, theta_grid, positive_floor)
    hi_x = _axis_value(right_p, log_grid, theta_grid, positive_floor)
    if hi_x < lo_x:
        lo_x, hi_x = hi_x, lo_x
    if hi_x == lo_x:
        return best_p, best_nll
    for _ in range(5):
        local_best_p = best_p
        local_best_nll = best_nll
        for j in range(201):
            x = lo_x + (hi_x - lo_x) * j / 200.0
            p = min(p_max, max(p_min, _axis_to_p(x, log_grid, theta_grid)))
            nll = negative_log_likelihood(p, rows)
            if nll < local_best_nll:
                local_best_nll = nll
                local_best_p = p
        best_p, best_nll = local_best_p, local_best_nll
        center = _axis_value(best_p, log_grid, theta_grid, positive_floor)
        width = (hi_x - lo_x) / 20.0
        lo_x = max(_axis_value(p_min, False, theta_grid, positive_floor) if not log_grid else _axis_value(max(p_min, positive_floor), True, False, positive_floor), center - width)
        hi_x = min(_axis_value(p_max, log_grid, theta_grid, positive_floor), center + width)
    return best_p, best_nll


def _low_p_suggestion(rows: List[Dict[str, int]], exact_p: Optional[float]) -> Dict[str, Any]:
    powers = [int(r["m"]) for r in rows]
    max_m = max(powers) if powers else 0
    from_power = 1.0 / float((2 * max_m + 1) ** 2) if max_m >= 0 else 1.0
    center = float(exact_p) if exact_p is not None and exact_p > 0.0 else from_power
    return {
        "suggested_p_min": 0.0,
        "suggested_p_max": min(1.0, max(center * 10.0, from_power * 10.0)),
        "basis": "exact_p" if exact_p is not None and exact_p > 0.0 else "largest_power_resolution",
    }


def fit_grid(
    rows: List[Dict[str, int]],
    grid_size: int = 20001,
    confidence: float = 0.95,
    exact_p: Optional[float] = None,
    keep_profile: bool = False,
    p_min: float = 0.0,
    p_max: float = 1.0,
    log_grid: bool = False,
    theta_grid: bool = False,
    detect_aliases: bool = False,
    expected_low_p: bool = False,
) -> Dict[str, Any]:
    # 1D grid/profile fit.  Low-probability AE should use p range constraints
    # and either log-grid or theta-grid to avoid selecting alias peaks.
    grid_size = max(101, int(grid_size))
    p_min = max(0.0, float(p_min))
    p_max = min(1.0, float(p_max))
    best_p = 0.0
    best_nll = math.inf
    profile: List[Tuple[float, float]] = []
    grid = _make_grid(p_min, p_max, grid_size, bool(log_grid), bool(theta_grid), exact_p)
    best_index = 0
    for i, p in enumerate(grid):
        nll = negative_log_likelihood(p, rows)
        profile.append((p, nll))
        if nll < best_nll:
            best_nll = nll
            best_p = p
            best_index = i

    best_p, best_nll = _refine_best(rows, profile, best_index, best_p, best_nll, p_min, p_max, bool(log_grid), bool(theta_grid))

    # Profile likelihood CI for one parameter: 2*DeltaNLL <= chi2_1.
    chi2_thresholds = {0.68: 1.0, 0.90: 2.705543, 0.95: 3.841459, 0.99: 6.634897}
    chi2 = chi2_thresholds.get(round(float(confidence), 2), 3.841459)
    delta_nll = 0.5 * chi2
    ci_points = [p for p, nll in profile if _finite(nll) and (nll - best_nll) <= delta_nll]
    ci = [min(ci_points), max(ci_points)] if ci_points else [best_p, best_p]
    local_minima = _detect_local_minima(profile, tolerance=delta_nll)
    alias_candidates = _detect_local_minima(profile, tolerance=None, max_count=25) if detect_aliases else []
    warnings = _warnings(
        rows,
        best_p,
        local_minima,
        alias_candidates=alias_candidates,
        p_min=p_min,
        p_max=p_max,
        log_grid=log_grid,
        theta_grid=theta_grid,
        expected_low_p=expected_low_p,
    )
    rel_dev = None
    exact_in_ci = None
    if exact_p is not None:
        exact_p = float(exact_p)
        rel_dev = None if exact_p == 0.0 else (float(best_p) - exact_p) / exact_p
        exact_in_ci = bool(ci[0] <= exact_p <= ci[1])

    total_oracle_calls = sum(int(r["shots"]) * (2 * int(r["m"]) + 1) for r in rows)
    grid_type = "theta" if theta_grid else "log_p" if log_grid else "linear_p"
    fit = {
        "fit_status": "ok",
        "method": f"{grid_type}_grid_profile_likelihood",
        "p_min": float(p_min),
        "p_max": float(p_max),
        "grid_size_requested": int(grid_size),
        "grid_points_evaluated": int(len(profile)),
        "grid_type": grid_type,
        "log_grid": bool(log_grid),
        "theta_grid": bool(theta_grid),
        "detect_aliases": bool(detect_aliases),
        "expected_low_p": bool(expected_low_p),
        "low_p_suggestion": _low_p_suggestion(rows, exact_p),
        "p_mle": float(best_p),
        "exact_p": None if exact_p is None else float(exact_p),
        "relative_deviation_vs_exact": rel_dev,
        "exact_in_confidence_interval": exact_in_ci,
        "confidence": float(confidence),
        "confidence_interval": [float(ci[0]), float(ci[1])],
        "negative_log_likelihood": float(best_nll),
        "fit_warning": warnings,
        "local_minima": local_minima,
        "alias_candidates": alias_candidates,
        "powers_used": sorted({int(r["m"]) for r in rows}),
        "total_shots": int(sum(int(r["shots"]) for r in rows)),
        "total_oracle_calls": int(total_oracle_calls),
        "rows_used": rows,
        "note": "CI is a simple one-parameter profile-likelihood interval on a finite grid. For rare AE, use p-range constraints with log-grid or theta-grid to avoid alias peaks.",
    }
    if keep_profile:
        fit["likelihood_profile"] = [
            {"p": float(p), "negative_log_likelihood": float(nll), "delta_nll": float(nll - best_nll)}
            for p, nll in profile
        ]
    return fit


def _write_csv(path: Path, fit: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "fit_status",
            "p_mle",
            "ci_low",
            "ci_high",
            "confidence",
            "exact_p",
            "exact_in_confidence_interval",
            "relative_deviation_vs_exact",
            "negative_log_likelihood",
            "fit_warning",
            "num_local_minima",
            "num_alias_candidates",
            "powers_used",
            "total_shots",
            "total_oracle_calls",
            "p_min",
            "p_max",
            "grid_type",
            "grid_points_evaluated",
            "method",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        ci = fit.get("confidence_interval", ["", ""])
        w.writerow(
            {
                "fit_status": fit.get("fit_status", ""),
                "p_mle": fit.get("p_mle", ""),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "confidence": fit.get("confidence", ""),
                "exact_p": fit.get("exact_p", ""),
                "exact_in_confidence_interval": fit.get("exact_in_confidence_interval", ""),
                "relative_deviation_vs_exact": fit.get("relative_deviation_vs_exact", ""),
                "negative_log_likelihood": fit.get("negative_log_likelihood", ""),
                "fit_warning": "|".join(str(x) for x in fit.get("fit_warning", [])),
                "num_local_minima": len(fit.get("local_minima", [])),
                "num_alias_candidates": len(fit.get("alias_candidates", [])),
                "powers_used": ",".join(str(x) for x in fit.get("powers_used", [])),
                "total_shots": fit.get("total_shots", ""),
                "total_oracle_calls": fit.get("total_oracle_calls", ""),
                "p_min": fit.get("p_min", ""),
                "p_max": fit.get("p_max", ""),
                "grid_type": fit.get("grid_type", ""),
                "grid_points_evaluated": fit.get("grid_points_evaluated", ""),
                "method": fit.get("method", ""),
            }
        )


def _write_profile_csv(path: Path, fit: Dict[str, Any]) -> None:
    profile = fit.get("likelihood_profile", [])
    if not profile:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["p", "negative_log_likelihood", "delta_nll"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in profile:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _json_ready_fit(fit: Dict[str, Any], profile_csv: Optional[Path] = None) -> Dict[str, Any]:
    out = dict(fit)
    profile = out.pop("likelihood_profile", None)
    if profile is not None:
        out["likelihood_profile_points"] = len(profile)
    if profile_csv is not None:
        out["likelihood_profile_csv"] = str(profile_csv)
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fit fixed-depth AE rows with a lightweight MLE.")
    ap.add_argument("--input", required=True, help="MLAE results.json or mlae_summary.csv.")
    ap.add_argument("--outdir", default=os.path.join("outputs", "extension", "scratch", "fixed_depth_ae_fit"))
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--grid-size", type=int, default=20001)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--exact-p", type=float, default=None)
    ap.add_argument("--p-min", type=float, default=0.0)
    ap.add_argument("--p-max", type=float, default=1.0)
    ap.add_argument("--log-grid", action="store_true")
    ap.add_argument("--theta-grid", action="store_true")
    ap.add_argument("--detect-aliases", action="store_true")
    ap.add_argument("--expected-low-p", action="store_true")
    ap.add_argument("--write-profile", action="store_true")
    ap.add_argument("--profile-csv", default="")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    rows = _to_fit_rows(_read_rows(Path(args.input)))
    fit = fit_grid(
        rows,
        grid_size=int(args.grid_size),
        confidence=float(args.confidence),
        exact_p=args.exact_p,
        keep_profile=bool(args.write_profile),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        log_grid=bool(args.log_grid),
        theta_grid=bool(args.theta_grid),
        detect_aliases=bool(args.detect_aliases),
        expected_low_p=bool(args.expected_low_p),
    )
    fit["input"] = str(args.input)

    outdir = Path(args.outdir)
    out_json = Path(args.out_json) if args.out_json else outdir / "fixed_depth_ae_fit.json"
    out_csv = Path(args.out_csv) if args.out_csv else outdir / "fixed_depth_ae_fit.csv"
    profile_csv = None
    if args.write_profile:
        profile_csv = Path(args.profile_csv) if args.profile_csv else outdir / "likelihood_profile.csv"
        _write_profile_csv(profile_csv, fit)
        print(f"Wrote {profile_csv}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(_json_ready_fit(fit, profile_csv), f, indent=2, ensure_ascii=False, allow_nan=False)
    _write_csv(out_csv, fit)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
