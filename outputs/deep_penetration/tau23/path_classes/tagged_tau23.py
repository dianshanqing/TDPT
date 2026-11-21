"""First-hit detection probabilities for the T=15 uniform absorbing slab."""
import argparse
from collections import defaultdict
from decimal import Decimal, localcontext
import json
from pathlib import Path


DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
UNCOLLIDED, FORWARD_ONLY, CHANGED_DIRECTION = range(3)
CLASSES = ("uncollided", "forward_scattered_only", "ever_changed_direction")
MODEL = {
    "grid_xyz": [17, 9, 9], "steps": 15, "groups": 1,
    "cell_width": 0.5, "slab_x_inclusive": [4, 11], "optical_depth": 23,
    "source_xyz": [2, 4, 4], "initial_direction": "+x", "initial_group": 0,
    "detector_x": 14, "detector_y_inclusive": [3, 5], "detector_z_inclusive": [3, 5],
    "slab_collision_probability": 0.9435838604962227,
    "absorption_given_collision": 0.998, "exterior_collision_probability": 0,
    "scattering_given_survival": "uniform over six Cartesian axis directions",
}
FROZEN_DETECTION = Decimal("1.0729208963459933e-10")


def calculate():
    """Propagate only live particle mass. Boundary hits and absorption terminate it."""
    with localcontext() as context:
        context.prec = 60
        return propagate()


def propagate():
    # The frozen configuration stored binary64 probabilities. Start from those exact values.
    collision = Decimal.from_float(MODEL["slab_collision_probability"])
    absorption = Decimal.from_float(MODEL["absorption_given_collision"])
    nx, ny, nz = MODEL["grid_xyz"]
    slab_first, slab_last = MODEL["slab_x_inclusive"]
    y_lo, y_hi = MODEL["detector_y_inclusive"]
    z_lo, z_hi = MODEL["detector_z_inclusive"]
    one = Decimal(1)
    zero = Decimal(0)
    live = {(*MODEL["source_xyz"], 0, UNCOLLIDED): one}
    detected = [zero] * len(CLASSES)
    first_hits = []
    for step in range(1, MODEL["steps"] + 1):
        next_live = defaultdict(Decimal)
        hits = [zero] * len(CLASSES)
        for (x, y, z, direction, tag), mass in live.items():
            # Termination tests precede the next flight, including on the final step.
            if x in (0, nx - 1) or y in (0, ny - 1) or z in (0, nz - 1):
                continue
            if x == MODEL["detector_x"] and y_lo <= y <= y_hi and z_lo <= z <= z_hi:
                hits[tag] += mass
                continue
            p_collision = collision if slab_first <= x <= slab_last else zero
            dx, dy, dz = DIRECTIONS[direction]
            next_live[x + dx, y + dy, z + dz, direction, tag] += mass * (one - p_collision)
            if p_collision:
                scattered_mass = mass * p_collision * (one - absorption) / len(DIRECTIONS)
                for new_direction, (dx, dy, dz) in enumerate(DIRECTIONS):
                    new_tag = FORWARD_ONLY
                    if direction != new_direction or tag == CHANGED_DIRECTION:
                        new_tag = CHANGED_DIRECTION
                    next_live[x + dx, y + dy, z + dz, new_direction, new_tag] += scattered_mass
        live = next_live
        detected = [old + new for old, new in zip(detected, hits)]
        if any(hits):
            first_hits.append([step, *map(str, hits)])
    total = sum(detected)
    n = slab_last - slab_first + 1
    uncollided = (one - collision) ** n
    forward_branch = collision * (one - absorption) / len(DIRECTIONS)
    forward_only = (one - collision + forward_branch) ** n - uncollided
    if max(abs(uncollided - detected[0]), abs(forward_only - detected[1])) >= Decimal("1e-64"):
        raise ArithmeticError("Tagged straight-path contributions disagree with the analytic check")
    relative_error = abs(total - FROZEN_DETECTION) / FROZEN_DETECTION
    if relative_error >= Decimal("1e-14"):
        raise ArithmeticError("Total detection probability disagrees with the frozen reference")
    return {
        "model": MODEL, "decimal_precision": 60, "class_names": CLASSES,
        "probabilities": list(map(str, detected)),
        "shares_percent": [str(value * 100 / total) for value in detected],
        "first_hit_columns": ["step", *CLASSES], "first_hit_table": first_hits,
        "sum": str(total), "frozen_classical": str(FROZEN_DETECTION),
        "relative_difference": str(relative_error), "uncollided_analytic": str(uncollided),
        "forward_scattered_only_analytic": str(forward_only),
        "exp_minus_tau": str(Decimal(-MODEL["optical_depth"]).exp()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="write computed JSON to this file")
    mode.add_argument("--check", action="store_true", help="compare with packaged results")
    args = parser.parse_args()
    output = json.dumps(calculate(), indent=2) + "\n"
    if args.check:
        expected = Path(__file__).with_name("tagged_tau23.json").read_text(encoding="utf-8")
        if json.loads(output) != json.loads(expected):
            raise SystemExit("Mismatch with tagged_tau23.json")
        print("All tagged probabilities and first-hit rows match.")
    elif args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
