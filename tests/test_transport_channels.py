"""Check sparse-basis X relabeling against the frozen evaluator without filtering."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iqae import OracleSpec, build_grover_operator_for_spec
from models import load_model
from reference_engine import evolve
from tools.verify_transport_channels import CHANNELS, INPUT, grover_step
from unitary_circuit import build_unitary_circuit_qiskit


class TransportMarkerTests(unittest.TestCase):
    def test_one_step_without_filtering(self):
        circuit, _ = build_unitary_circuit_qiskit(load_model(INPUT), 1, kernel_profile="baseline")
        regs = {r.name: r for r in circuit.qregs}
        for target in CHANNELS:
            with self.subTest(target=target):
                q, _, _ = build_grover_operator_for_spec(
                    A_circuit=circuit, status_qubits=regs["status"], g_qubits=regs["g"],
                    spec=OracleSpec(good=target))
                initial, _ = evolve(q.state_preparation)
                expected, info = evolve(q, initial)
                observed, run = grover_step(q, initial, cutoff=0, max_states=300000)
                self.assertEqual(observed, expected)
                self.assertEqual(run["primitive_gates"], info["primitive_gates"])
                self.assertEqual(run["discarded_l2_sum"], 0)


if __name__ == "__main__":
    unittest.main()
