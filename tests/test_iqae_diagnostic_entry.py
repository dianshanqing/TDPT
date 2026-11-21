"""Fixed-power diagnostic requests must never dispatch an estimator implicitly."""
import unittest
from pathlib import Path
from run_request import resolve_run_request

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticEntryTests(unittest.TestCase):
    def request(self, **values):
        return resolve_run_request(dict(task='iqae', config='q_uniform_small',
            steps=3, cpu_budget=1, **values), project_root=ROOT)

    def test_compile_is_finite_and_explicit(self):
        r = self.request(iqae_diagnostic='compile', iqae_powers='0,6',
                         seed=2026110803, bond=128, mps_trunc=1e-16)
        self.assertEqual(r['arguments']['iqae_powers'], [0, 6])
        self.assertEqual(r['diagnostic_contract']['mode'], 'compile')
        self.assertFalse(r['diagnostic_contract']['sampling_executed'])
        self.assertFalse(r['diagnostic_contract']['estimator_executed'])
        self.assertEqual(r['arguments']['seed'], 2026110803)
        self.assertEqual(r['arguments']['iqae_max_compiled_gates'], 4000000)

    def test_incomplete_or_invalid_requests_fail(self):
        for values in [dict(iqae_powers='6'), dict(iqae_diagnostic='compile'),
                       dict(iqae_diagnostic='compile', iqae_powers='-1'),
                       dict(iqae_diagnostic='compile', iqae_powers='0,0'),
                       dict(iqae_diagnostic='compile', iqae_powers='6', iqae_max_grover_power=5),
                       dict(iqae_diagnostic='compile', iqae_powers='0', iqae_max_compiled_gates=0)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.request(**values)

    def test_no_effect_on_estimator_or_direct_mps(self):
        r = self.request()
        self.assertIsNone(r['diagnostic_contract'])
        self.assertEqual(r['precision_contract']['relative_error'], .01)
        self.assertEqual(r['precision_contract']['confidence'], .95)
        with self.assertRaises(ValueError):
            resolve_run_request(dict(task='mps_probe', suite='baseline_tau23_probe',
                iqae_diagnostic='compile', iqae_powers='0'), project_root=ROOT)


if __name__ == '__main__':
    unittest.main()
