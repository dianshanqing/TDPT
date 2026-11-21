"""Reduced response kernel (RRK) helpers for rare-response validation."""

from .model import (
    RRKResult,
    ae_power_probability,
    build_rrk_grover,
    build_rrk_state_preparation,
    exact_probability,
    infer_eta,
    run_rrk_analytic_fixed_depth,
    run_rrk_fixed_depth,
    run_rrk_mc,
)

__all__ = [
    "RRKResult",
    "ae_power_probability",
    "build_rrk_grover",
    "build_rrk_state_preparation",
    "exact_probability",
    "infer_eta",
    "run_rrk_analytic_fixed_depth",
    "run_rrk_fixed_depth",
    "run_rrk_mc",
]
