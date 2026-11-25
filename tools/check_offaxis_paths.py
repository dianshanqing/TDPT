"""Independent shortest-path reference for the five-step, single-cell duct detector."""

import argparse
import hashlib
import itertools
import json
from decimal import Decimal, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/models/q_duct_offaxis.json"
AXES = ("+x", "+y", "+z")


def enumerate_paths(config):
    geom = config["geom"]
    src = config["source"]
    start = (src["x0"], src["y0"], src["z0"])
    target = (geom["detector_x"], geom["roi_y0"], geom["roi_z0"])
    shape = (geom["nx"], geom["ny"], geom["nz"])
    if (config["model"], shape, start, target, src["dir0"], src["g0"]) != (
        "legacy_duct", (5, 4, 4), (1, 1, 1), (3, 2, 2), 0, 0
    ):
        raise ValueError("This path reference requires the specified off-axis duct and +x source.")
    if (geom["roi_y1"], geom["roi_z1"]) != target[1:] or config["groups"]["n_groups"] != 1:
        raise ValueError("The detector must cover one cell and the model must use one energy group.")
    displacement = [a - b for a, b in zip(target, start)]
    shortest = tuple(axis for axis, count in enumerate(displacement) for _ in range(count))
    rows = []
    by_turn = {}
    for sequence in sorted(set(itertools.permutations(shortest))):
        pos = list(start)
        incoming = 0
        weight = Decimal(1)
        turns = 0
        factors = []
        for outgoing in sequence:
            x, y, z = pos
            material = "concrete"
            if geom["slab_x0"] <= x <= geom["slab_x1"]:
                duct = geom["duct_y0"] <= y <= geom["duct_y1"] and geom["duct_z0"] <= z <= geom["duct_z1"]
                material = "air" if duct else "tungsten"
            entry = config["mats"][material]
            p_int = Decimal(entry["p_int"][0])
            p_abs = Decimal(entry["p_abs"][0])
            scattered = p_int * (1 - p_abs) / 6
            changed = incoming != outgoing
            factor = scattered if changed else 1 - p_int + scattered
            zero_reason = None
            if factor == 0:
                if changed and p_int == 0:
                    zero_reason = "The outgoing direction changes in a zero-interaction cell."
                else:
                    zero_reason = "The configured transition has zero probability."
            factors.append(dict(position=pos.copy(), material_slot=material, incoming=AXES[incoming],
                                outgoing=AXES[outgoing], weight=str(factor), zero_reason=zero_reason))
            weight *= factor
            turns += changed
            pos[outgoing] += 1
            if not all(0 < coord < size - 1 for coord, size in zip(pos, shape)):
                raise ValueError("A shortest path touches the boundary before detection.")
            incoming = outgoing
        if tuple(pos) != target:
            raise ValueError("Shortest path did not reach the target.")
        by_turn[turns] = by_turn.get(turns, Decimal(0)) + weight
        rows.append(dict(moves=[AXES[axis] for axis in sequence], turns_from_source_heading=turns,
                         factors=factors, detection_probability=str(weight)))
    total = sum(by_turn.values())
    return dict(displacement=displacement, earliest_scoring_step=len(shortest) + 1,
                p_detect_through_steps_1_to_4=["0"] * len(shortest),
                p_detect_through_step_5=str(total),
                p_detect_by_turn_count={str(n): str(p) for n, p in sorted(by_turn.items())},
                shortest_direction_sequences=len(rows),
                positive_weight_sequences=sum(Decimal(row["detection_probability"]) > 0 for row in rows),
                paths=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True, help="new JSON output file")
    args = parser.parse_args()
    data = args.config.read_bytes()
    with localcontext() as ctx:
        ctx.prec = 50
        result = enumerate_paths(json.loads(data, parse_float=Decimal))
    result.update(config=args.config.name, config_sha256=hashlib.sha256(data).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arithmetic="50 significant decimal digits, using the JSON decimal inputs",
                  scoring="A detector arrival after the fourth move registers at the start of step five.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(result["p_detect_through_step_5"])


if __name__ == "__main__":
    main()
