"""Independent integer predicates and explicit quantum-capability failures."""
from dataclasses import replace
from itertools import product
import math
import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from config import default_config, paper_scale_config, validate_config
import unitary_circuit as uc


def permute_basis(circuit, basis):
    """Execute only actual X/controlled-X gates; fail if another gate appears."""
    for instruction in circuit.data:
        op = instruction.operation
        base = getattr(op, "base_gate", op)
        if base.name != "x":
            raise AssertionError("Non-permutation gate in Boolean predicate: " + op.name)
        bits = [circuit.find_bit(q).index for q in instruction.qubits]
        count = getattr(op, "num_ctrl_qubits", 0)
        pattern = getattr(op, "ctrl_state", (1 << count) - 1)
        if all(((basis >> bit) & 1) == ((pattern >> j) & 1) for j, bit in enumerate(bits[:count])):
            basis ^= 1 << bits[-1]
    return basis


class TestGeometryPredicates(unittest.TestCase):
    def test_all_small_grid_points_single_adjacent_wide_and_full_intervals(self):
        """Catches endpoint XOR cancellation and omission, with reversible cleanup."""
        base = default_config()
        intervals = [(lo, hi) * 5 for lo, hi in ((1, 1), (1, 2), (0, 2), (0, 3))]
        # Asymmetric y/z and slab/duct predicates must not share the wrong bounds.
        intervals.append((0, 3, 1, 1, 2, 3, 0, 2, 2, 2))
        for sx0, sx1, dy0, dy1, dz0, dz1, ry0, ry1, rz0, rz1 in intervals:
            cfg = replace(base, geom=replace(base.geom, nx=4, ny=4, nz=4,
                slab_x0=sx0, slab_x1=sx1, duct_y0=dy0, duct_y1=dy1, duct_z0=dz0, duct_z1=dz1,
                detector_x=1, roi_y0=ry0, roi_y1=ry1, roi_z0=rz0, roi_z1=rz1),
                source=replace(base.source, x0=1, y0=1, z0=1))
            x, y, z = (QuantumRegister(2, n) for n in ("x", "y", "z"))
            work = QuantumRegister(10, "work")
            roi = QuantumCircuit(x, y, z, work)
            roi_bit = uc.compile_roi_predicate(roi, x, y, z, list(work)[:9], cfg)
            roi_inverse = QuantumCircuit(x, y, z, work)
            uc.uncompile_roi_predicate(roi_inverse, x, y, z, list(work)[:9], cfg)
            material = QuantumCircuit(x, y, z, work)
            slab_bit, duct_bit = uc.compile_material_predicates(material, x, y, z, list(work), cfg)
            material_inverse = QuantumCircuit(x, y, z, work)
            uc.uncompile_material_predicates(material_inverse, x, y, z, list(work), cfg)
            for a, b, c in product(range(4), repeat=3):
                with self.subTest(geometry=cfg.geom, point=(a, b, c)):
                    original = a | (b << 2) | (c << 4)
                    state = permute_basis(roi, original)
                    self.assertEqual((state >> roi.find_bit(roi_bit).index) & 1,
                        int(a == 1 and ry0 <= b <= ry1 and rz0 <= c <= rz1))
                    self.assertEqual(permute_basis(roi_inverse, state), original)
                    state = permute_basis(material, original)
                    self.assertEqual((state >> material.find_bit(slab_bit).index) & 1, int(sx0 <= a <= sx1))
                    self.assertEqual((state >> material.find_bit(duct_bit).index) & 1, int(dy0 <= b <= dy1 and dz0 <= c <= dz1))
                    self.assertEqual(permute_basis(material_inverse, state), original)


class TestQuantumCapabilities(unittest.TestCase):
    def test_four_groups_remain_valid_classically_but_fail_quantum(self):
        cfg = paper_scale_config()
        validate_config(cfg)
        with self.assertRaises(ValueError):
            uc.build_unitary_circuit_qiskit(cfg, 1, kernel_profile="compressed")

    def test_reachable_downscatter_is_not_silently_ignored(self):
        cfg = default_config()
        cfg = replace(cfg, mats=replace(cfg.mats, concrete=replace(cfg.mats.concrete, q_down=(.2, 0))))
        validate_config(cfg)
        with self.assertRaises(ValueError):
            uc.build_unitary_circuit_qiskit(cfg, 1, kernel_profile="compressed")

    def test_invalid_steps_positions_ranges_and_nonfinite_probabilities(self):
        cfg = default_config()
        for steps in (0, -1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                uc.build_unitary_circuit_qiskit(cfg, steps, kernel_profile="compressed")
        bad = [replace(cfg, source=replace(cfg.source, x0=1.5)),
            replace(cfg, source=replace(cfg.source, dir0=6)),
            replace(cfg, geom=replace(cfg.geom, roi_y0=4, roi_y1=3)),
            replace(cfg, geom=replace(cfg.geom, slab_x1=16)),
            replace(cfg, geom=replace(cfg.geom, nx=16.5))]
        for value in (math.nan, math.inf, -.1, 1.1):
            bad.append(replace(cfg, mats=replace(cfg.mats,
                concrete=replace(cfg.mats.concrete, p_int=(value, .12)))))
        for index, case in enumerate(bad):
            with self.subTest(case=index), self.assertRaises(ValueError):
                uc.build_unitary_circuit_qiskit(case, 1, kernel_profile="compressed")

    def test_unsafe_manual_and_unproved_disable_flags_are_rejected(self):
        cfg = default_config()
        for kwargs in (dict(kernel_profile="manual"),
            dict(kernel_profile="manual", use_dir_tape=True, use_absorb_tape=True),
            dict(kernel_profile="compressed", disable_boundary=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                uc.build_unitary_circuit_qiskit(cfg, 1, **kwargs)

    def test_tiny_positive_rotation_is_not_dropped(self):
        circuit = QuantumCircuit(2)
        circuit.x(0)
        uc._multi_controlled_ry(circuit, 2e-20, [0], 1)
        state = Statevector.from_instruction(circuit)
        self.assertGreater(abs(state.data[3]), 0)
        self.assertAlmostEqual(abs(state.data[3]) / 1e-20, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
