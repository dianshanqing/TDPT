"""Scientific entrypoint contracts for configuration, isolation and resources."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RevisionEntrypointTests(unittest.TestCase):
    def test_actual_circuit_reports_raw_and_compiled_resources(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs', prefix='revision_circuit_') as temp:
            out = Path(temp)/'circuit'
            done = subprocess.run([sys.executable, '-B', str(ROOT/'main.py'),
                '--phase', 'circuit', '--config', 'configs/models/q_uniform_small.json',
                '--steps', '1', '--aer-safe-basis', '--outdir', str(out)],
                cwd=temp, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout+done.stderr)
            result = json.loads((out/'results.json').read_text(encoding='utf8'))
            self.assertGreater(result['resources']['compiled_gates'], 0)
            self.assertGreater(result['resources']['compiled_two_qubit_gates'], 0)
            self.assertEqual(len(result['register_map']['status']), 2)

    def test_new_phases_parse_and_no_default_wallclock(self):
        import main
        for phase in ('screen', 'reference', 'validate', 'mps_probe'):
            args = main.build_argparser().parse_args(['--phase', phase])
            self.assertIsNone(args.run_timeout_sec)
            self.assertIsNone(args.iqae_max_call_sec)
            self.assertEqual(args.cpu_budget, 4)

    def test_main_import_does_not_load_quantum_modules(self):
        code = "import sys; import main; assert 'qiskit' not in sys.modules; assert 'unitary_circuit' not in sys.modules"
        done = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_model_and_steps_have_distinct_identity(self):
        from config import default_config
        from runtime_identity import make_identity
        def ident(cfg, steps):
            return make_identity(config=cfg, profile=None, compile_options=None,
                measurement=None, backend=None, phase='reference', extra={'steps':steps})['identity_hash']
        cfg = default_config()
        self.assertNotEqual(ident(cfg, 2), ident(cfg, 3))
        self.assertNotEqual(ident(cfg, 2), ident(replace(cfg, source=replace(cfg.source, x0=2)), 2))

    def test_resource_budget_rejects_oversubscription(self):
        import main
        args = main.build_argparser().parse_args(['--cpu-budget', '2', '--aer-threads', '3'])
        with self.assertRaisesRegex(ValueError, 'CPU budget'):
            main._configure_cpu_budget(args)

    def test_reference_runs_from_other_directory_without_quantum(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs', prefix='revision_entry_') as temp:
            out = Path(temp)/'reference'
            code = (
                "import sys, importlib.abc; "
                f"sys.path.insert(0, {str(ROOT)!r}); "
                "exec(\"class Block(importlib.abc.MetaPathFinder):\\n"
                " def find_spec(self, fullname, path=None, target=None):\\n"
                "  if fullname.startswith(('qiskit', 'unitary_circuit')): raise ImportError('quantum forbidden')\"); "
                "sys.meta_path.insert(0,Block()); import main; "
                # Current reference requires an explicit model; legacy-reference is a separate alias.
                f"raise SystemExit(main.main(['--phase','reference','--config','configs/models/q_legacy_small.json','--steps','2','--outdir',{str(out)!r}]))"
            )
            done = subprocess.run([sys.executable, '-B', '-c', code], cwd=temp,
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout+done.stderr)
            result = json.loads((out/'result.json').read_text(encoding='utf8'))
            self.assertEqual(result['steps'], 2)
            self.assertAlmostEqual(sum(result['rows'][-1]['probabilities']), 1)


if __name__ == '__main__':
    unittest.main()
