import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_request import resolve_run_request

ROOT = Path(__file__).resolve().parents[1]
SUITE = 'configs/suites/compact_cloud_tau23_b128.json'


class RequestTests(unittest.TestCase):
    def resolve(self, **kw):
        return resolve_run_request(kw, project_root=ROOT)

    def test_cmd_shards_forward_one_effective_parameter_set(self):
        import run, re
        from unittest.mock import patch
        env=dict(re.findall(r'^set "([^=]+)=(.*)"$', (ROOT/'run.cmd').read_text(), re.M))
        env['VRQO_RUN_CMD']='1'
        env['VRQO_CMD_REQUEST_JSON']='{"shots": 999999}'
        calls=[]
        with patch.dict('os.environ',env,clear=True),patch.object(run,'require_qc'),patch.object(run,'invoke',side_effect=lambda path,args: calls.append((path,args)) or 0):
            self.assertEqual(run.main([]),0)
        path,args=calls[0]
        self.assertEqual(path,'shard_runner.py')
        for flag,value in (('--num-shards','50'),('--max-parallel','50'),('--total-shots','15000'),('--shot-chunk','300'),('--seed0','1234'),('--steps','15')):
            self.assertEqual(args[args.index(flag)+1],value)
            self.assertEqual(args.count(flag),1)
        self.assertIn('--dry-run',args)
        extra=json.loads(args[args.index('--extra-json')+1])
        for flag,value in (('--mps-max-bond','896'),('--mps-trunc','1e-16'),('--run-memory-mib','98304'),('--aer-heartbeat-sec','60'),('--resource-sample-sec','1'),('--poll-sec','0.1'),('--blas-threads','1')):
            self.assertEqual(extra[extra.index(flag)+1],value)
            self.assertEqual(extra.count(flag),1)
        self.assertIn('--test-measure-x',extra)
        self.assertIn('--test-measure-dir',extra)
        self.assertIn('--aer-safe-basis',extra)
        self.assertNotIn('--debug-measure-work',extra)
        self.assertNotIn('--shots',extra)

    def test_cmd_shards_cli_override_and_paths_with_spaces(self):
        import run
        from unittest.mock import patch
        env=dict(VRQO_RUN_CMD='1',TASK='test',SHARD_MODE='1',CONFIG='legacy_duct',SHOTS='15000',NUM_SHARDS='50',MAX_PARALLEL='50',SHOT_CHUNK='300',DRY_RUN='1',OUTDIR='outputs/path with spaces')
        calls=[]
        with patch.dict('os.environ',env,clear=True),patch.object(run,'require_qc'),patch.object(run,'invoke',side_effect=lambda path,args: calls.append((path,args)) or 0):
            run.main(['--shots','12','--num-shards','3','--max-parallel','3','--shot-chunk','4','--config','q_uniform_small'])
        args=calls[0][1]
        self.assertEqual(args[args.index('--total-shots')+1],'12')
        self.assertEqual(args[args.index('--num-shards')+1],'3')
        self.assertTrue(args[args.index('--outroot')+1].endswith('outputs\\path with spaces'))
        extra=json.loads(args[args.index('--extra-json')+1])
        self.assertEqual(extra[extra.index('--config')+1],'q_uniform_small')
        self.assertEqual(extra.count('--config'),1)

    def test_single_mode_preserves_existing_main_cli_and_execute_override(self):
        import run
        from unittest.mock import patch
        env=dict(VRQO_RUN_CMD='1',TASK='test',SHARD_MODE='1',CONFIG='q_uniform_small',DRY_RUN='1')
        calls=[]
        with patch.dict('os.environ',env,clear=True),patch.object(run,'require_qc'),patch.object(run,'invoke',side_effect=lambda path,args: calls.append((path,args)) or 0):
            run.main(['--shard-mode','0','--num-shards','2','--shard-id','1','--execute'])
        path,args=calls[0]
        self.assertEqual(path,'main.py')
        self.assertEqual(args[args.index('--num-shards')+1],'2')
        self.assertIn('--execute',args)
        self.assertNotIn('--shard-mode',args)

    def test_mc_histories_reach_main_from_cmd_and_cli_override(self):
        import main
        from unittest.mock import patch
        cmd=dict(task='mc',config='q_uniform_small',steps=2,histories='37')
        for flags,expected in ((['--phase','mc'],37),(['--phase','mc','--histories','51'],51)):
            args=main.build_argparser().parse_args(flags)
            with patch.dict('os.environ',{'VRQO_CMD_REQUEST_JSON':json.dumps(cmd)}):
                resolved=main._resolve_request(args,flags)
            self.assertEqual(args.histories,expected)
            self.assertEqual(resolved['arguments']['histories'],expected)
        for value in (0,-1,True,'bad'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.resolve(task='mc',config='q_uniform_small',histories=value)
        with self.assertRaisesRegex(ValueError,'inapplicable'):
            self.resolve(task='iqae',config='q_mps_one_cell',histories=100)

    def test_cmd_mode_filter_and_blank_sampling_defaults(self):
        from run import _cmd_request
        from unittest.mock import patch
        env={'VRQO_RUN_CMD':'1','TASK':'iqae','CONFIG':'q_mps_one_cell','STEPS':'2',
             'HISTORIES':'91','SHOTS':'','COUNTS_CHECK':'0','BATCH_WORKERS':'2',
             'RELATIVE_ERROR':'.01','CONFIDENCE':'.95'}
        with patch.dict('os.environ',env,clear=True):
            selected=_cmd_request('iqae',False)
            self.assertNotIn('histories',selected)
            self.assertNotIn('counts_check',selected)
            self.assertNotIn('batch_workers',selected)
            self.assertEqual(self.resolve(cmd=selected)['arguments']['shots'],512)
            self.assertEqual(_cmd_request('reference',True),{})

    def test_summary_distinguishes_missing_zero_and_numerical_certification(self):
        from run import summarize_probe
        from unittest.mock import patch
        rows=[dict(status='complete',probabilities=None,reference=[0.,1e-10,1.-1e-10,0.]),
              dict(status='complete',probabilities=[0.,0.,1.,0.],reference=[0.,1e-10,1.-1e-10,0.],work_any_nonzero_union_upper_bound=0.),
              dict(status='complete',probabilities=[0.,1e-10,1.-1e-10,0.],reference=[0.,1e-10,1.-1e-10,0.],work_any_nonzero_union_upper_bound=.001)]
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary,patch('runtime_identity.PROJECT_ROOT',Path(temporary)):
            out=Path(temporary)/'outputs';out.mkdir()
            source=out/'input.json';source.write_text(json.dumps({'rows':rows}))
            summarize_probe(source,out/'summary')
            reported=json.loads((out/'summary'/'result.json').read_text())['rows']
        self.assertIsNone(reported[0]['p_D'])
        self.assertEqual(reported[0]['rare_accuracy_status'],'not_assessed')
        self.assertEqual(reported[1]['relative_error'],1.)
        self.assertEqual(reported[1]['rare_accuracy_status'],'failed')
        self.assertEqual(reported[2]['rare_accuracy_status'],'passed')
        self.assertEqual(reported[2]['numerical_validity_status'],'unresolved')
        self.assertFalse(reported[2]['scientific_certification_passed'])

    def test_production_defaults_and_relative_precision_are_distinct(self):
        got=self.resolve(task='iqae',config='q_mps_one_cell',steps=1)
        self.assertEqual(got['arguments']['profile'],'baseline')
        self.assertEqual(got['arguments']['relative_error'],.01)
        self.assertEqual(got['arguments']['confidence'],.95)
        self.assertEqual(got['arguments']['iqae_max_shots'],200000)
        self.assertEqual(got['precision_contract']['scope'],'single_case_complete_estimation')
        custom=self.resolve(task='iqae',config='q_mps_one_cell',relative_error=.02,
            confidence=.99,iqae_max_shots=800,iqae_max_queries=9000,iqae_max_grover_power=16,
            bond=64,mps_trunc=1e-15)
        self.assertEqual(custom['arguments']['relative_error'],.02)
        self.assertEqual(custom['arguments']['bond'],64)
        self.assertEqual(custom['arguments']['iqae_max_grover_power'],16)
        # Explicit historical profiles retain their circuit meaning.
        old=self.resolve(task='iqae',config='q_mps_one_cell',profile='compressed')
        self.assertEqual(old['arguments']['profile'],'compressed')

    def test_invalid_relative_precision_and_budgets_are_rejected(self):
        for key, value in [('relative_error',0),('relative_error',float('nan')),
                ('confidence',1),('confidence',0),('confidence',True),
                ('iqae_max_shots',0),('iqae_max_queries',True),('iqae_max_grover_power',-1)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                self.resolve(task='iqae',config='q_mps_one_cell',**{key:value})
        with self.assertRaisesRegex(ValueError,'inapplicable'):
            self.resolve(task='mps_probe',config='q_mps_one_cell',confidence=.95)

    def test_suite_defaults_and_no_mutation(self):
        request = dict(task='mps_probe', suite=SUITE)
        before = copy.deepcopy(request)
        got = resolve_run_request(request, project_root=ROOT)
        self.assertEqual(request, before)
        self.assertEqual(got['cases'][0]['steps'], 15)
        self.assertEqual(got['backend']['bond_dimensions'], [128])
        self.assertEqual(got['arguments']['seed'], 20261030)
        self.assertTrue(got['arguments']['counts_check'])

    def test_cli_over_cmd_over_suite(self):
        got = self.resolve(cmd=dict(task='mps_probe', suite=SUITE, steps=12, bond=256),
                           cli=dict(steps=14, bond=64, seed=71, counts_check=False))
        self.assertEqual(got['cases'][0]['steps'], 14)
        self.assertEqual(got['effective_suite']['bond_dimensions'], [64])
        self.assertEqual(got['effective_suite']['seed'], 71)
        self.assertFalse(got['effective_suite']['counts_check'])
        self.assertEqual(got['provenance']['steps'], 'cli')
        self.assertTrue(got['overrides'])

    def test_model_steps_and_absolute_path_other_cwd(self):
        got = self.resolve(task='reference', config='configs/models/compact_deep/compact_n8_w0002_tau23.json')
        self.assertEqual(got['arguments']['config'],'compact_n8_w0002_tau23')
        self.assertEqual(got['cases'][0]['model_document']['model'], 'uniform_slab')

    def test_multicase_needs_selection(self):
        with self.assertRaisesRegex(ValueError, 'CASE_ID|case_id'):
            self.resolve(task='validate', suite='configs/suites/validation_small.json', steps=1)
        got = self.resolve(task='validate', suite='configs/suites/validation_small.json', case_id='q_uniform_small', steps=1, profile='compressed')
        self.assertEqual(len(got['cases']), 1)
        self.assertEqual(got['effective_suite']['cases'][0]['profiles'], ['compressed'])

    def test_rejects_bad_requests(self):
        bad = [dict(task='reference', config='a', suite=SUITE),
               dict(task='reference', bond=128), dict(task='reference', mystery=1),
               dict(task='mps_probe', suite=SUITE, case_id='missing'),
               dict(task='mps_probe', suite=SUITE, steps=0),
               dict(task='mps_probe', suite=SUITE, bond=True),
               dict(task='mps_probe', suite=SUITE, mps_trunc=float('nan')),
               dict(task='mps_probe', suite=SUITE, counts_check='maybe'),
               dict(task='mps_probe', suite=SUITE, cpu_budget=2, aer_threads=2, mps_threads=2)]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(ValueError):
                resolve_run_request(request, project_root=ROOT)

    def test_dry_run_resolves_without_creating_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
            out = Path(d)/'not_created'
            got = self.resolve(task='mps_probe', suite=SUITE, outdir=str(out), dry_run=True)
            self.assertFalse(out.exists())
            self.assertTrue(got['arguments']['dry_run'])

    def test_unknown_nested_field(self):
        with self.assertRaises(ValueError):
            self.resolve(cli=dict(task='reference', typo=4))

    def test_single_config_probe_and_cmd_steps_win_model(self):
        got=self.resolve(task='mps_probe', config='configs/models/q_mps_one_cell.json', steps=1, bond=64)
        self.assertEqual(got['effective_suite']['cases'][0]['steps'],1)
        self.assertEqual(got['effective_suite']['bond_dimensions'],[64])
        import main
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
            out=Path(d)/'dry'
            cmd=dict(config='configs/models/q_mps_one_cell.json',steps=1,outdir=str(out),dry_run=True)
            import contextlib,io
            captured=io.StringIO()
            with patch.dict('os.environ',{'VRQO_CMD_REQUEST_JSON':json.dumps(cmd)}),contextlib.redirect_stdout(captured):
                self.assertEqual(main.main(['--phase','reference']),0)
            r=json.loads(captured.getvalue())['request']
            self.assertEqual(r['arguments']['steps'],1)
            self.assertFalse(out.exists())

    def test_dry_run_no_aer_import_in_fresh_process(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
            out=Path(d)/'dry'
            code="import sys,main; main.main(['--phase','mps_probe','--suite',"+repr(SUITE)+",'--dry-run','--outdir',"+repr(str(out))+"]); assert 'qiskit_aer' not in sys.modules"
            p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_reference_needs_explicit_model_and_manual_flags_survive(self):
        with self.assertRaisesRegex(ValueError,'CONFIG|config'):
            self.resolve(task='reference')
        import main
        from unittest.mock import patch
        flags=['--phase','circuit','--full-history-mode','--dir-tape','--absorb-tape','--term-tape','--scatter-tape']
        args=main.build_argparser().parse_args(flags)
        with patch.dict('os.environ',{'VRQO_CMD_REQUEST_JSON':'{}'}):
            main._resolve_request(args,flags)
        profile=main._resolved_kernel_profile(args)
        self.assertEqual(profile.name,'manual')
        self.assertTrue(profile.full_history_mode)
        self.assertEqual(args.resolved_request['arguments']['profile'],'manual')

    def test_inapplicable_legacy_flags_and_bad_backend_rejected_before_run(self):
        import main
        from unittest.mock import patch
        for flag in (['--transpiler-seed','71'],['--aer-max-mem-mb','64'],['--xt-no-interaction']):
            original=['--phase','mps_probe','--suite',SUITE,*flag]
            args=main.build_argparser().parse_args(original)
            with patch.dict('os.environ',{'VRQO_CMD_REQUEST_JSON':'{}'}),self.assertRaises(ValueError):
                main._resolve_request(args,original)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
            from models import load_document
            spec=load_document(SUITE)
            spec['backend_options']['mps_lapac']=True
            path=Path(d)/'bad.json';path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError): self.resolve(task='mps_probe',suite=str(path),dry_run=True)


if __name__ == '__main__':
    unittest.main()
