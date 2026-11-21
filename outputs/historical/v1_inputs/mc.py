"""Classical Monte Carlo baseline for the transport benchmark."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import math

from config import TransportConfig, validate_config
from geometry import is_boundary, is_in_roi, material_at, clamp_pos

_DIRS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]

@dataclass
class MCResult:

    p_detect: float
    p_absorb: float
    p_boundary: float
    p_survive: float

    se_detect: float
    se_absorb: float
    se_boundary: float
    se_survive: float

    energy_hist_detect: np.ndarray
    counts: Dict[str, int]


@dataclass
class RareMCResult:

    eta: float
    p_detect_reach_hat: float
    p_rare_analog_hat: float
    p_rare_weighted_hat: float

    se_detect_reach: float
    se_rare_analog: float
    se_rare_weighted: float

    detect_reach_count: int
    analog_rare_count: int
    histories: int
    naive_MC_required_histories: Dict[str, float]


def _naive_required_histories(p: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    p = float(p)
    for rel_error in (0.5, 0.2, 0.1):
        key = f"R={rel_error:g}"
        if p <= 0.0:
            out[key] = math.inf
        else:
            out[key] = max((1.0 - p) / ((rel_error ** 2) * p), 0.0)
    return out


def run_mc(
    cfg: TransportConfig,
    n_steps: int,
    histories: int,
    seed: Optional[int] = 1234,
    unitary_matched_energy: bool = False,
) -> MCResult:

    validate_config(cfg)
    rng = np.random.default_rng(seed)

    G = cfg.groups.n_groups
    hist_detect = np.zeros(G, dtype=np.int64)

    c_detect = 0
    c_absorb = 0
    c_boundary = 0
    c_survive = 0

    for _ in range(histories):
        x = cfg.source.x0
        y = cfg.source.y0
        z = cfg.source.z0
        g = cfg.source.g0
        dir_idx = int(getattr(cfg.source, "dir0", 0))

        status = "alive"

        for _t in range(n_steps):

            if is_boundary(cfg, x, y, z):
                status = "boundary"
                c_boundary += 1
                break

            if is_in_roi(cfg, x, y, z):
                status = "detect"
                c_detect += 1
                hist_detect[g] += 1
                break

            mat = material_at(cfg, x, y, z)
            mp = getattr(cfg.mats, mat)
            p_int = float(mp.p_int[g])
            p_abs = float(mp.p_abs[g])
            q_down = float(mp.q_down[g])

            if rng.random() < p_int:

                if rng.random() < p_abs:
                    status = "absorb"
                    c_absorb += 1
                    break

                dir_idx = int(rng.integers(0, 6))

                if g < G - 1 and rng.random() < q_down:
                    if unitary_matched_energy:
                        g = (g + 1) % G
                    else:
                        g = min(g + 1, G - 1)

            dx, dy, dz = _DIRS[dir_idx]
            x, y, z = clamp_pos(cfg, x + dx, y + dy, z + dz)

        else:
            status = "survive"
            c_survive += 1

    p_detect = c_detect / histories
    p_absorb = c_absorb / histories
    p_boundary = c_boundary / histories
    p_survive = c_survive / histories

    se_detect = math.sqrt(max(p_detect * (1 - p_detect), 0.0) / histories)
    se_absorb = math.sqrt(max(p_absorb * (1 - p_absorb), 0.0) / histories)
    se_boundary = math.sqrt(max(p_boundary * (1 - p_boundary), 0.0) / histories)
    se_survive = math.sqrt(max(p_survive * (1 - p_survive), 0.0) / histories)

    return MCResult(
        p_detect=p_detect,
        p_absorb=p_absorb,
        p_boundary=p_boundary,
        p_survive=p_survive,
        se_detect=se_detect,
        se_absorb=se_absorb,
        se_boundary=se_boundary,
        se_survive=se_survive,
        energy_hist_detect=hist_detect.astype(np.float64) / max(c_detect, 1),
        counts=dict(detect=c_detect, absorb=c_absorb, boundary=c_boundary, survive=c_survive),
    )


def run_mc_rare_response(
    cfg: TransportConfig,
    n_steps: int,
    histories: int,
    eta: float,
    seed: Optional[int] = 1234,
    unitary_matched_energy: bool = False,
) -> RareMCResult:

    validate_config(cfg)
    if histories <= 0:
        raise ValueError("histories must be positive")
    eta = float(eta)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0, 1] for analog rare-response sampling")

    rng = np.random.default_rng(seed)
    G = cfg.groups.n_groups

    c_detect = 0
    c_rare_analog = 0

    for _ in range(histories):
        x = cfg.source.x0
        y = cfg.source.y0
        z = cfg.source.z0
        g = cfg.source.g0
        dir_idx = int(getattr(cfg.source, "dir0", 0))

        for _t in range(n_steps):

            if is_boundary(cfg, x, y, z):
                break

            if is_in_roi(cfg, x, y, z):
                c_detect += 1
                if rng.random() < eta:
                    c_rare_analog += 1
                break

            mat = material_at(cfg, x, y, z)
            mp = getattr(cfg.mats, mat)
            p_int = float(mp.p_int[g])
            p_abs = float(mp.p_abs[g])
            q_down = float(mp.q_down[g])

            if rng.random() < p_int:

                if rng.random() < p_abs:
                    break

                dir_idx = int(rng.integers(0, 6))

                if g < G - 1 and rng.random() < q_down:
                    if unitary_matched_energy:
                        g = (g + 1) % G
                    else:
                        g = min(g + 1, G - 1)

            dx, dy, dz = _DIRS[dir_idx]
            x, y, z = clamp_pos(cfg, x + dx, y + dy, z + dz)

    p_detect = c_detect / histories
    p_rare_analog = c_rare_analog / histories
    p_rare_weighted = eta * p_detect

    se_detect = math.sqrt(max(p_detect * (1.0 - p_detect), 0.0) / histories)
    se_analog = math.sqrt(max(p_rare_analog * (1.0 - p_rare_analog), 0.0) / histories)
    se_weighted = abs(eta) * se_detect

    return RareMCResult(
        eta=eta,
        p_detect_reach_hat=p_detect,
        p_rare_analog_hat=p_rare_analog,
        p_rare_weighted_hat=p_rare_weighted,
        se_detect_reach=se_detect,
        se_rare_analog=se_analog,
        se_rare_weighted=se_weighted,
        detect_reach_count=int(c_detect),
        analog_rare_count=int(c_rare_analog),
        histories=int(histories),
        naive_MC_required_histories=_naive_required_histories(p_rare_weighted),
    )
