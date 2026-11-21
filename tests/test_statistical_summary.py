"""Short statistical/report-contract tests; no quantum simulation is submitted."""
import copy
import csv
from dataclasses import asdict
import io
import contextlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import run
import stats_utils
from config import default_config
from runtime_identity import make_identity, normalize

ROOT = Path(__file__).resolve().parents[1]


class StatisticalSummaryTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('VRQO_OUTPUT_DIR', ROOT/'outputs/local_validation/20261028_run_control_01/stats_tests'))
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.config = normalize(asdict(default_config()))
        identity = make_identity(config=self.config, profile=None, compile_options=None,
                                 measurement={'terminal_counts':['survive','detect','absorb','boundary']},
                                 backend={'method':'numpy_mc'}, phase='mc', extra={'steps':2})
        self.raw = {'phase':'mc','status':'complete','run_id':'statistical-fixture','request_hash':identity['identity_hash'],
                    'identity':identity,'steps':2,'histories':10,'counts':{'survive':4,'detect':3,'absorb':2,'boundary':1},
                    'sampling':{'sample_start':0,'sample_stop':10,'total_samples':10,'num_shards':1,
                                'shard_id':0,'root_seed':71,'actual_seed':71,'seed_strategy':'root_seed + shard_id'}}
        self.raw_path = self.save(self.raw, 'raw.json')
        with contextlib.redirect_stdout(io.StringIO()):
            run.reference(['--steps','2','--outdir',str(self.folder/'reference')])
        self.reference_path = self.folder/'reference/result.json'
        self.reference = json.loads(self.reference_path.read_text())

    def save(self, value, name):
        path = self.folder/name
        path.write_text(json.dumps(value), encoding='utf8')
        return path

    def summarize(self, *extra, name='summary', raw=None):
        with contextlib.redirect_stdout(io.StringIO()):
            code = run.summarize(['--input',str(raw or self.raw_path),'--outdir',str(self.folder/name),*map(str,extra)])
        self.assertEqual(code,0)
        return json.loads((self.folder/name/'result.json').read_text())

    def test_default_summary_reports_four_simultaneous_intervals_without_certification(self):
        result = self.summarize()
        self.assertIs(result.get('precision_certified'),False)
        report = result['sampling_statistics']
        self.assertEqual(report['family_size'],4)
        self.assertEqual(report['confidence'],.95)
        self.assertEqual(report['samples'],10)
        self.assertEqual(set(report['channels']),set(self.raw['counts']))
        self.assertNotIn('tolerance',report)

    def test_reference_contains_explicit_configuration_and_completed_status(self):
        self.assertEqual(self.reference['config'],self.config)
        self.assertEqual(self.reference.get('status'),'complete')
        self.assertEqual(self.reference['rows'][-1]['steps'],2)

    def test_current_main_reference_identity_and_model_are_supported(self):
        from reference_solver import reference
        from models import load_model
        cfg=load_model(ROOT/'configs/models/q_layered_small.json')
        record=reference(cfg,2)
        record.update(phase='reference',status='complete',identity=make_identity(config=cfg,profile=None,
            compile_options=None,measurement=None,backend={'method':'independent_float64_recurrence'},phase='reference',extra={'steps':2}))
        raw=copy.deepcopy(self.raw)
        raw['identity']['config']=normalize(asdict(cfg))
        path=self.save(record,'current_reference.json')
        probabilities,provenance=run.reference_probabilities(path,raw)
        self.assertEqual(probabilities['detect'],record['rows'][-1]['p_D'])
        self.assertEqual(provenance['phase'],'reference')
        record['identity']['config']['h']*=2
        with self.assertRaises(ValueError): run.reference_probabilities(self.save(record,'tampered_reference.json'),raw)

    def test_eight_member_family_and_reference_differences_are_written_to_json_and_csv(self):
        result = self.summarize('--reference',self.reference_path,'--family-size','8')
        report = result['sampling_statistics']
        self.assertEqual(report['family_size'],8)
        row = report['channels']['detect']
        self.assertEqual(row['reference'],0.0)
        self.assertEqual(row['signed_difference'],.3)
        self.assertEqual(row['difference_interval'],row['confidence_interval'])
        self.assertIn('sha256',report['reference_record'])
        with (self.folder/'summary/statistics.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),4)
        self.assertEqual({int(r['family_size']) for r in rows},{8})
        self.assertTrue(all(r['precision_certified']=='False' for r in rows))
        self.assertEqual(sum(int(r['count']) for r in rows),10)

    def test_reference_channel_order_is_read_and_alive_alias_maps_to_survive(self):
        ref=copy.deepcopy(self.reference)
        old=ref['order'];new=['boundary','alive','absorb','detect']
        for row in ref['rows']:
            lookup=dict(zip(old,row['probabilities']))
            row['probabilities']=[lookup['survive' if x=='alive' else x] for x in new]
        ref['order']=new
        result=self.summarize('--reference',self.save(ref,'permuted.json'))
        rows=result['sampling_statistics']['channels']
        expected=dict(zip(self.reference['order'],self.reference['rows'][-1]['probabilities']))
        for terminal in self.raw['counts']:self.assertEqual(rows[terminal]['reference'],expected[terminal])

    def test_reference_rejects_missing_mismatched_or_invalid_model_and_rows(self):
        for case in ['missing_config','config_mismatch','missing_step','duplicate_step','fractional_step',
                     'wrong_phase','failed_reference','duplicate_channel','not_normalized','nan','negative','bad_norm_error']:
            ref=copy.deepcopy(self.reference)
            if case=='missing_config':ref.pop('config')
            if case=='config_mismatch':ref['config']['geom']['detector_x']=11
            if case=='missing_step':ref['rows']=ref['rows'][:1]
            if case=='duplicate_step':ref['rows'].append(copy.deepcopy(ref['rows'][-1]))
            if case=='fractional_step':ref['rows'][-1]['steps']=2.0
            if case=='wrong_phase':ref['phase']='mc'
            if case=='failed_reference':ref['status']='failed'
            if case=='duplicate_channel':ref['order']=['alive','survive','absorb','boundary']
            if case=='not_normalized':ref['rows'][-1]['probabilities'][0]=.5
            if case=='nan':ref['rows'][-1]['probabilities'][0]=float('nan')
            if case=='negative':ref['rows'][-1]['probabilities'][0]=-.1
            if case=='bad_norm_error':ref['rows'][-1]['norm_error']=.1
            with self.subTest(case=case),self.assertRaises(ValueError):
                self.summarize('--reference',self.save(ref,case+'.json'),name=case)
            self.assertFalse((self.folder/case).exists())

    def test_raw_missing_counts_failed_status_or_identity_step_disagreement_is_rejected(self):
        for case in ['counts','failed','steps']:
            raw=copy.deepcopy(self.raw)
            if case=='counts':raw.pop('counts')
            if case=='failed':raw['status']='failed'
            if case=='steps':raw['steps']=1
            with self.subTest(case=case),self.assertRaises(ValueError):
                self.summarize('--reference',self.reference_path,name=case,raw=self.save(raw,case+'.json'))

    def test_cp_zero_and_center_are_exact_and_4096_does_not_imply_one_percent(self):
        self.assertTrue(hasattr(stats_utils,'sampling_intervals'))
        report=stats_utils.sampling_intervals({'survive':2048,'detect':2048,'absorb':0,'boundary':0},family_size=8)
        self.assertAlmostEqual(report['channels']['absorb']['confidence_interval'][1],1-(.05/16)**(1/4096),places=14)
        center=report['channels']['detect']['confidence_interval']
        self.assertAlmostEqual(center[0],.4785268796534659,places=12)
        self.assertAlmostEqual(center[1],.521473120346534,places=12)
        self.assertGreater(center[1]-.5,.01)
        self.assertIs(report['precision_certified'],False)

    def test_interval_helper_rejects_fractional_counts_invalid_family_and_empty_samples(self):
        self.assertTrue(hasattr(stats_utils,'sampling_intervals'))
        for counts,m in [({'a':.5,'b':.5},2),({'a':0},1),({'a':1,'b':1},1),({'a':1},True)]:
            with self.subTest(counts=counts,m=m),self.assertRaises(ValueError):
                stats_utils.sampling_intervals(counts,family_size=m)


if __name__=='__main__':unittest.main()
