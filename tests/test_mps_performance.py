"""Finite prefix probes must keep performance and scientific certification apart."""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models import load_document


class PrefixPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mps_performance'),
                             'the finite-prefix performance entry point must exist')
        self.module=importlib.import_module('mps_performance')
        self.temp=tempfile.TemporaryDirectory(prefix='vrqo_prefix_test_')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch('runtime_identity.PROJECT_ROOT',self.root).start()
        # Progress is an actual guarded write. Its inherited supervisor path
        # belongs to the outer runtime, so isolate it with this fixture's root.
        patch.dict(os.environ,{'VRQO_PROGRESS_FILE':str(
            self.root/'outputs'/'_fixture_progress'/'progress.json')}).start()
        # Keep a stable, nonempty source identity while other agents edit entries.
        patch('runtime_identity.source_manifest',return_value={'fixture.py':'1'*64}).start()
        self.spec={'cases':[{'id':'tiny_prefix','config':'q_mps_one_cell','steps':2,
                             'profile':'compressed'}],
                   'bond_dimensions':[4],'truncation_thresholds':[1e-16],
                   'counts_check':False,'seed':123,
                   'backend_options':{'precision':'double','zero_threshold':0.,'chop_threshold':0.}}

    @staticmethod
    def tiny_circuit(*args,**kwargs):
        from qiskit import QuantumCircuit,QuantumRegister
        status=QuantumRegister(2,'status');work=QuantumRegister(1,'work')
        circuit=QuantumCircuit(status,work,global_phase=.17)
        circuit.h(status[0]);circuit.cx(status[0],work[0])
        # Both later gates depend on the CX; opt0 may reorder independent gates.
        circuit.x(status[0]);circuit.rx(.2,work[0])
        return circuit,{'fixture':'four-gate prefix contract'}

    def run_probe(self,name='run',*,spec=None,prefix=2,reference=None,**kwargs):
        with patch('unitary_circuit.build_unitary_circuit_qiskit',side_effect=self.tiny_circuit):
            return self.module.run_prefix_probe(spec or self.spec,self.root/'outputs'/name,
                kwargs.pop('cpu_budget',2),prefix_gates=prefix,reference_path=reference,**kwargs)

    def raw(self,name='run'):
        return json.loads((self.root/'outputs'/name/'prefix_result.json').read_text())

    def test_actual_prefix_preserves_phase_registers_and_only_one_verbose_log(self):
        report=self.run_probe()
        row=self.raw()
        self.assertEqual(row['execution_status'],'passed')
        self.assertTrue(report['passed'])
        self.assertEqual(row['science_status'],'not_assessed_prefix')
        self.assertEqual(row['comparison']['status'],'reference_only')
        self.assertEqual(row['normalization_status'],'passed')
        self.assertEqual(row['prefix_gates'],2)
        self.assertEqual(row['full_compiled_gates'],4)
        self.assertEqual(row['register_map'],{'status':[0,1],'work':[2]})
        self.assertAlmostEqual(row['prefix_global_phase'],.17)
        for got,want in zip(row['probabilities'],[.5,.5,0.,0.]):
            self.assertAlmostEqual(got,want,places=13)
        self.assertAlmostEqual(row['work_bit_probabilities'][0],.5,places=13)
        self.assertNotEqual(row['prefix_fingerprint'],row['compiled_A_fingerprint'])
        self.assertNotEqual(row['diagnostic_fingerprint'],row['prefix_fingerprint'])
        # This fixture runs in a shared unittest process. Aer keeps previous
        # MPS logs globally, so only a fresh main.py worker can attribute a cap.
        self.assertIsNone(row['bond_statistics']['max_observed_bond'])
        self.assertGreaterEqual(row['bond_statistics']['max_log_bond'],2)
        self.assertFalse(row['bond_statistics']['current_run_attribution_verified'])
        self.assertFalse(row['bond_statistics']['representative_bond_near_cap'])
        self.assertFalse(row['bond_statistics']['reached_cap'])
        self.assertNotIn('MPS_log_data',report['metadata'])
        self.assertIn('MPS_log_data',row['metadata'])
        ref=report['metadata_log_references']['MPS_log_data']
        self.assertEqual(ref['path'],'prefix_result.json')
        path=self.root/'outputs/run/prefix_result.json'
        self.assertEqual(ref['file_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(report,json.loads((path.parent/'performance_summary.json').read_text()))
        self.assertEqual({p.name for p in path.parent.iterdir()},
                         {'prefix_result.json','performance_summary.json'})
        for stage in ('build','compile','fingerprints','evolution_and_reads','result_extraction','write'):
            self.assertGreaterEqual(report['stages'][stage]['wall_seconds'],0.)
            self.assertIn('tree_cpu_seconds',report['stages'][stage])
        self.assertIn('exited',report['cpu_measurement_limitations'])
        self.assertIsNone(report['comparison']['evolution_speedup'])

    def test_equal_thread_variant_compares_to_summary_without_claiming_full_science(self):
        self.run_probe('baseline')
        reference=self.root/'outputs/baseline/performance_summary.json'
        report=self.run_probe('thread2',reference=reference,mps_threads=2)
        self.assertEqual(report['comparison']['status'],'passed')
        self.assertTrue(report['comparison']['comparable'])
        self.assertLessEqual(report['comparison']['max_abs_observable_error'],2e-11)
        self.assertGreater(report['comparison']['evolution_speedup'],0.)
        self.assertEqual(report['comparison']['observed_evolution_speed_ratio'],report['comparison']['evolution_speedup'])
        self.assertTrue(report['comparison']['speedup_accepted'])
        self.assertEqual(report['formal_D_accuracy_status'],'not_assessed_prefix')
        self.assertFalse(report['comparison']['baseline_scientific_accuracy_certified'])
        self.assertEqual(report['science_status'],'not_assessed_prefix')

    def test_single_real_transport_prefix_never_uses_final_dp_reference(self):
        with patch('reference_solver.reference',side_effect=AssertionError('full DP is not a prefix reference')):
            report=self.module.run_prefix_probe(self.spec,self.root/'outputs/transport',2,prefix_gates=60)
        self.assertEqual(report['execution_status'],'passed')
        self.assertGreater(report['full_compiled_gates'],60)
        self.assertEqual(report['science_status'],'not_assessed_prefix')
        self.assertNotIn('reference',report)

    def test_real_transport_blas_and_lapack_controls_preserve_small_prefix(self):
        from numerical_runtime import configure_blas_threads
        self.addCleanup(configure_blas_threads, 1)
        baseline=self.module.run_prefix_probe(self.spec,self.root/'outputs/real_baseline',8,
            prefix_gates=60,mps_threads=1,blas_threads=1)
        self.assertEqual(baseline['normalization_status'],'passed')
        reference=self.root/'outputs/real_baseline/prefix_result.json'
        blas=self.module.run_prefix_probe(self.spec,self.root/'outputs/real_blas8',8,
            prefix_gates=60,mps_threads=1,blas_threads=8,reference_path=reference)
        self.assertEqual(blas['comparison']['status'],'passed')
        aer=[r for r in blas['numerical_libraries_after'] if r['owner']=='aer_private']
        self.assertTrue(aer)
        self.assertTrue(all(r['num_threads']==8 for r in aer))
        spec=copy.deepcopy(self.spec)
        spec['backend_options']['mps_lapack']=True
        lapack=self.module.run_prefix_probe(spec,self.root/'outputs/real_lapack',8,
            prefix_gates=60,mps_threads=1,blas_threads=1,reference_path=reference)
        self.assertEqual(lapack['comparison']['status'],'passed')
        self.assertIs(lapack['backend']['effective']['mps_lapack'],True)
        # This checks wiring and probabilities only, not representative speedup.
        self.assertFalse(lapack['comparison']['baseline_scientific_accuracy_certified'])

    def test_invalid_scan_counts_and_budget_rejected_before_backend_or_output(self):
        invalid=[]
        for key,value in [('cases',self.spec['cases']*2),('bond_dimensions',[2,4]),
                          ('truncation_thresholds',[0.,1e-16]),('counts_check',True),
                          ('counts_check',0),('backend_options',{'mps_log_data':False})]:
            spec=copy.deepcopy(self.spec);spec[key]=value;invalid.append(spec)
        for spec in invalid:
            with self.subTest(spec=spec),self.assertRaises(ValueError):self.run_probe(spec=spec)
        for prefix in (0,-1,True,1.5):
            with self.subTest(prefix=prefix),self.assertRaises(ValueError):self.run_probe(prefix=prefix)
        with self.assertRaises(ValueError):self.run_probe(cpu_budget=1,mps_threads=2)
        self.assertFalse((self.root/'outputs/run').exists())

    def test_full_or_longer_prefix_never_evolves_and_records_failure(self):
        from qiskit_aer import AerSimulator
        for prefix in (4,5):
            with self.subTest(prefix=prefix),patch.object(AerSimulator,'run',side_effect=AssertionError('must not evolve full A')):
                report=self.run_probe('full'+str(prefix),prefix=prefix)
            self.assertEqual(report['execution_status'],'failed')
            self.assertIn('strictly shorter',report['error'])
            self.assertIsNone(report['probabilities'])

    def test_backend_failure_is_preserved_without_zero_fill(self):
        from qiskit_aer import AerSimulator
        with patch.object(AerSimulator,'run',side_effect=RuntimeError('controlled backend failure')):
            report=self.run_probe()
        self.assertEqual(report['execution_status'],'failed')
        self.assertFalse(report['passed'])
        self.assertIn('controlled backend failure',report['error'])
        self.assertIsNone(report['probabilities'])
        self.assertIsNone(report['work_bit_probabilities'])
        self.assertEqual(report['comparison']['status'],'not_evaluated')

    def test_unknown_aer_blas_library_cannot_claim_effective_configuration(self):
        with patch('numerical_runtime.configure_blas_threads',return_value=[]):
            report=self.run_probe()
        self.assertEqual(report['execution_status'],'failed')
        self.assertIn('Aer BLAS',report['error'])
        self.assertIsNone(report['probabilities'])

    def test_incompatible_reference_rejected_before_evolution(self):
        self.run_probe('baseline')
        path=self.root/'outputs/baseline/prefix_result.json'
        baseline=self.raw('baseline')
        from qiskit_aer import AerSimulator
        for key,value in [('prefix_fingerprint','0'*64),('seed',987),('bond',8),
                          ('config',{'wrong':'model'}),('diagnostic_fingerprint','f'*64)]:
            changed=copy.deepcopy(baseline);changed[key]=value
            reference=self.root/(key+'.json');reference.write_text(json.dumps(changed))
            with self.subTest(key=key),patch.object(AerSimulator,'run',side_effect=AssertionError('incompatible reference must not evolve')):
                report=self.run_probe('bad_'+key,reference=reference)
            self.assertEqual(report['execution_status'],'failed')
            self.assertIn('reference mismatch',report['error'])
            self.assertIsNone(report['probabilities'])
        self.assertTrue(path.exists())

    def test_reference_sources_versions_or_observable_layout_must_match(self):
        self.run_probe('baseline');baseline=self.raw('baseline')
        from runtime_identity import digest
        for key,value in [('sources',{'different.py':'f'*64}),('versions',{'python':'0.0'}),
                          ('measurement',{'status_order':['detect','survive','absorb','boundary']})]:
            row=copy.deepcopy(baseline);row['identity'][key]=value
            row['identity']['identity_hash']=digest({k:v for k,v in row['identity'].items() if k!='identity_hash'})
            reference=self.root/(key+'.json');reference.write_text(json.dumps(row))
            report=self.run_probe('bad_'+key,reference=reference)
            self.assertEqual(report['execution_status'],'failed')
            self.assertIn('reference mismatch',report['error'])

    def test_reference_error_and_bad_normalization_do_not_offer_speedup(self):
        self.run_probe('baseline');baseline=self.raw('baseline')
        changed=copy.deepcopy(baseline);changed['probabilities'][0]-=1e-4;changed['probabilities'][1]+=1e-4
        reference=self.root/'different_observable.json';reference.write_text(json.dumps(changed))
        report=self.run_probe('different',reference=reference)
        self.assertEqual(report['execution_status'],'passed')
        self.assertEqual(report['comparison']['status'],'failed')
        self.assertGreater(report['comparison']['max_abs_observable_error'],2e-11)
        self.assertIsNone(report['comparison']['evolution_speedup'])
        changed=copy.deepcopy(baseline);changed['probabilities'][0]+=1e-5
        reference=self.root/'bad_norm.json';reference.write_text(json.dumps(changed))
        report=self.run_probe('bad_norm',reference=reference)
        self.assertEqual(report['comparison']['status'],'failed')
        self.assertGreater(report['comparison']['reference_norm_residual'],2e-11)
        self.assertIsNone(report['comparison']['evolution_speedup'])
        self.assertGreater(report['comparison']['observed_evolution_speed_ratio'],0.)
        self.assertFalse(report['comparison']['speedup_accepted'])
        self.assertEqual(report['formal_D_accuracy_status'],'not_assessed_prefix')

    def test_overwrite_refused_before_new_execution(self):
        self.run_probe();old=(self.root/'outputs/run/prefix_result.json').read_bytes()
        with self.assertRaises(FileExistsError):self.run_probe()
        self.assertEqual((self.root/'outputs/run/prefix_result.json').read_bytes(),old)

    def test_nested_thread_product_cannot_exceed_total_budget(self):
        with self.assertRaises(ValueError):
            self.run_probe(cpu_budget=50,mps_threads=8,blas_threads=8)
        self.assertFalse((self.root/'outputs/run').exists())

    def test_numerical_library_after_snapshot_is_taken_after_the_actual_backend_run(self):
        import numerical_runtime
        from qiskit_aer import AerSimulator
        original_libraries=numerical_runtime.numerical_libraries
        original_run=AerSimulator.run;has_run=False
        def libraries():
            return [dict(record,snapshot_after_run=has_run) for record in original_libraries()]
        def run(sim,*args,**kwargs):
            nonlocal has_run
            result=original_run(sim,*args,**kwargs);has_run=True;return result
        with patch('numerical_runtime.numerical_libraries',side_effect=libraries),patch.object(AerSimulator,'run',new=run):
            report=self.run_probe()
        self.assertTrue(all(item['snapshot_after_run'] for item in report['numerical_libraries_after']))
        self.assertTrue(all(not item['snapshot_after_run'] for item in report['numerical_libraries_configured']))

    def test_current_normalization_failure_is_not_a_passing_equivalence(self):
        self.run_probe('baseline')
        from qiskit_aer import AerSimulator
        from types import SimpleNamespace
        original_run=AerSimulator.run
        def nonnormalized(sim,*args,**kwargs):
            result=original_run(sim,*args,**kwargs).result()
            result.data()['status_probability'][0]+=1e-5
            return SimpleNamespace(result=lambda:result)
        with patch.object(AerSimulator,'run',new=nonnormalized):
            report=self.run_probe('bad_current',reference=self.root/'outputs/baseline/prefix_result.json')
        self.assertEqual(report['execution_status'],'passed')
        self.assertEqual(report['normalization_status'],'failed')
        self.assertEqual(report['comparison']['status'],'failed')
        self.assertIsNone(report['comparison']['evolution_speedup'])

    def test_different_host_rejected_even_with_same_circuit_and_versions(self):
        self.run_probe('baseline')
        baseline=self.raw('baseline');baseline['host_identity']['host_name']='another-host'
        path=self.root/'different_host.json';path.write_text(json.dumps(baseline))
        report=self.run_probe('other_host',reference=path)
        self.assertEqual(report['execution_status'],'failed')
        self.assertIn('reference mismatch',report['error'])
        self.assertIsNone(report['probabilities'])

    def test_optional_readouts_do_not_assume_intermediate_work_is_zero(self):
        spec=copy.deepcopy(self.spec);spec['readout_diagnostics']=True
        report=self.run_probe(spec=spec)
        self.assertEqual(report['execution_status'],'passed')
        self.assertEqual(report['readout_diagnostics']['consistency_status'],'passed')
        self.assertEqual(report['work_accuracy_status'],'not_applicable_intermediate_compiled_prefix')
        self.assertAlmostEqual(report['work_bit_probabilities'][0],.5)
        self.assertIn('host_identity',report)
        self.assertFalse(any('audit_mps_before' in key for key in report))
        self.assertFalse(report['readout_diagnostics']['tensor_snapshot_enabled'])
        self.assertIsNone(report['readout_diagnostics']['tensor_norm_before'])

    def test_multiple_thread_controls_cannot_claim_a_controlled_speedup(self):
        self.run_probe('baseline',cpu_budget=4)
        report=self.run_probe('two_changes',cpu_budget=4,mps_threads=2,blas_threads=2,
            reference=self.root/'outputs/baseline/prefix_result.json')
        self.assertEqual(report['execution_status'],'passed')
        self.assertEqual(report['comparison']['numerical_comparison_status'],'passed')
        self.assertFalse(report['comparison']['one_factor_controlled'])
        self.assertEqual(len(report['comparison']['changed_controls']),2)
        self.assertIsNone(report['comparison']['evolution_speedup'])
        self.assertEqual(report['comparison']['status'],'failed')


if __name__=='__main__':unittest.main()
