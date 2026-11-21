"""Replay a two-cell sparse IQAE run from its recorded counts in the QC environment."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from iqae import estimate_relative_iqae


def replay(path):
    record = json.loads(path.read_bytes())
    protocol = record["protocol"]["protocol"]
    batches = iter(record["measurement_batches"])

    def counts(power, shots):
        batch = next(batches)
        if (power, shots) != (batch["power"], batch["shots"]):
            raise ValueError(f"Power or shot allocation differs at power {power}")
        return batch["success_count"]

    result = estimate_relative_iqae(
        counts,
        relative_error=protocol["relative_error"],
        confidence=protocol["confidence"],
        shots=protocol["shots_per_round"],
        max_shots=protocol["max_shots"],
        max_queries=protocol["max_queries"],
        max_power=protocol["max_power"],
    )
    if next(batches, None) is not None:
        raise ValueError("Replay stopped before using every recorded batch")
    if result != record["estimator"]:
        raise ValueError("Replayed estimator differs from the recorded estimator")
    lower, upper = result["last_valid_interval"]
    print(f"{result['status']}, interval [{lower:.16g}, {upper:.16g}]")
    print(f"Relative midpoint bound {(upper - lower) / (2 * lower):.6%}")
    print(f"Shots {result['costs']['shots']}, A and inverse calls {result['costs']['queries_A_and_inverse']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="result.json of the original two-cell sparse run")
    replay(parser.parse_args().result)
