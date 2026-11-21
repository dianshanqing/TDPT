"""Physical-model regression tests, evolved from actual unitary instructions."""
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time
import unittest

from config import default_config
from unitary_circuit import build_unitary_circuit_qiskit, resolve_kernel_profile
from runtime_identity import circuit_fingerprint
from reference_engine import evolve, marginal, reference, build_traced, observe, config_with_probs

PROFILES = ("baseline", "compressed", "full_history")
TOL = 2e-11


def run_case(cfg, steps, profile, traced=False):
    if traced:
        circuit, _, checkpoints = build_traced(cfg, steps, profile)
    else:
        circuit, _ = build_unitary_circuit_qiskit(cfg, steps, kernel_profile=profile)
        checkpoints = None
    fingerprint_start = time.perf_counter()
    fingerprint = circuit_fingerprint(circuit)
    fingerprint_seconds = time.perf_counter() - fingerprint_start
    state, runtime = evolve(circuit, timeout=None, max_states=300000,
        checkpoints=checkpoints, observer=observe if traced else None)
    got = marginal(circuit, state, ["status", "x", "y", "z", "dir", "g"])
    want = reference(cfg, steps, retain_joint=True)["joint"]
    joint_error = max((abs(got.get(k, 0) - want.get(k, 0)) for k in got.keys() | want.keys()), default=0)
    work = math.fsum(p for (value,), p in marginal(circuit, state, ["work"]).items() if value)
    bad_direction = math.fsum(p for (s, x, y, z, d, g), p in got.items() if s == 1 and d != 0)
    report = dict(profile=profile, actual_profile=asdict(resolve_kernel_profile(kernel_profile=profile)),
        config=asdict(cfg), steps=steps, qubits=circuit.num_qubits, top_level_gates=len(circuit.data),
        circuit_fingerprint=fingerprint, fingerprint_seconds=fingerprint_seconds,
        probability_absolute_tolerance=TOL,
        physical_joint_columns=["status", "x", "y", "z", "dir", "g"],
        work_observation={"register": "work", "indices": list(range(17)), "complete": True},
        max_joint_error=joint_error,
        work_nonzero=work, detect_non_plus_x=bad_direction,
        norm=math.fsum(got.values()), runtime=runtime)
    compact={key:report[key] for key in ('profile','steps','qubits','circuit_fingerprint',
        'max_joint_error','work_nonzero','detect_non_plus_x','norm','probability_absolute_tolerance')}
    compact['physical_step_checkpoints']=[{'step':row.get('step'),
        'work_nonzero':row['observables']['work_nonzero'],
        'norm':row['observables']['norm']} for row in runtime['checkpoints']]
    if os.environ.get('VRQO_OUTPUT_DIR'):
        with (Path(os.environ['VRQO_OUTPUT_DIR'])/'transport_science.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(compact)+'\n')
    print(json.dumps(compact), flush=True)
    return report


class TestTransportRegression(unittest.TestCase):
    def test_detected_real_prefix_freezes_direction_and_cleans_work(self):
        """Catches the old tD-as-alive defect; old T3 gives .068625/.00915."""
        cfg = default_config()
        cfg = replace(cfg, geom=replace(cfg.geom, detector_x=2))
        for profile in PROFILES:
            with self.subTest(profile=profile):
                result = run_case(cfg, 3, profile)
                self.assertLess(result["work_nonzero"], TOL, result)
                self.assertLess(result["max_joint_error"], TOL, result)

    def test_dangerous_manual_shared_term_is_rejected_before_simulation(self):
        """Record the old real-prefix resurrection if capability rejection is absent."""
        cfg = default_config()
        cfg = replace(cfg, geom=replace(cfg.geom, detector_x=0), source=replace(cfg.source, x0=0))
        try:
            circuit, _ = build_unitary_circuit_qiskit(cfg, 2, kernel_profile="manual")
        except ValueError:
            return
        state, runtime = evolve(circuit, timeout=None, max_states=2000)
        joint = marginal(circuit, state, ["status", "x", "y", "z", "dir"])
        result = {"case": "manual_shared_term", "joint": [list(k) + [v] for k, v in joint.items()], "runtime": runtime}
        print(json.dumps(result), flush=True)
        self.fail("Dangerous manual configuration accepted: " + json.dumps(result))


class TestTransportModel(unittest.TestCase):
    def assert_model(self, cfg, steps, profile, traced=False):
        result = run_case(cfg, steps, profile, traced=traced)
        self.assertLess(abs(result["norm"] - 1), TOL, result)
        self.assertLess(result["max_joint_error"], TOL, result)
        self.assertLess(result["work_nonzero"], TOL, result)
        for row in result["runtime"]["checkpoints"]:
            self.assertLess(row["observables"]["work_nonzero"], TOL, row)

    def test_three_profiles_default_shallow_joint_distribution(self):
        for profile in PROFILES:
            for steps in (1, 2, 3):
                with self.subTest(profile=profile, steps=steps):
                    self.assert_model(default_config(), steps, profile)

    def test_three_post_detection_steps_preserve_every_physical_variable(self):
        cfg = default_config()
        cfg = replace(cfg, geom=replace(cfg.geom, detector_x=2))
        for profile in PROFILES:
            for steps in (4, 5):
                with self.subTest(profile=profile, steps=steps):
                    self.assert_model(cfg, steps, profile, traced=True)

    def test_boundary_priority_repeated_absorb_and_short_horizon(self):
        cfg = default_config()
        overlap = replace(cfg, geom=replace(cfg.geom, detector_x=0), source=replace(cfg.source, x0=0))
        absorbed = config_with_probs(cfg, 1, 1)
        no_collision = config_with_probs(cfg, 0, 0)
        for profile in PROFILES:
            for name, case in (("overlap", overlap), ("absorb", absorbed), ("short", no_collision)):
                with self.subTest(profile=profile, case=name):
                    self.assert_model(case, 3, profile, traced=True)

    def test_non_degenerate_random_histories_and_general_geometry(self):
        cfg = default_config()
        cases = [config_with_probs(cfg, .37, .23), config_with_probs(cfg, 1, 0)]
        cases += [replace(cfg, geom=replace(cfg.geom, detector_x=2, roi_y0=3, roi_y1=3, roi_z0=3, roi_z1=3)),
            replace(cfg, geom=replace(cfg.geom, detector_x=2, roi_y0=1, roi_y1=6, roi_z0=1, roi_z1=6,
                slab_x0=1, slab_x1=4, duct_y0=2, duct_y1=5, duct_z0=2, duct_z1=5))]
        one_group_mats = replace(cfg.mats, **{name: replace(getattr(cfg.mats, name),
            p_int=getattr(cfg.mats, name).p_int[:1], p_abs=getattr(cfg.mats, name).p_abs[:1],
            q_down=(0.0,)) for name in ("concrete", "tungsten", "air")})
        cases.append(replace(cfg, groups=replace(cfg.groups, n_groups=1, boundaries=(1.0, .01)), mats=one_group_mats))
        last_group_mats = replace(cfg.mats, **{name: replace(getattr(cfg.mats, name),
            q_down=(0.0, .5)) for name in ("concrete", "tungsten", "air")})
        cases.append(replace(cfg, source=replace(cfg.source, g0=1), mats=last_group_mats))
        for profile in PROFILES:
            for index, case in enumerate(cases):
                with self.subTest(profile=profile, case=index):
                    self.assert_model(case, 3, profile)


if __name__ == "__main__":
    unittest.main()
