"""Finite real-circuit diagnostics, exact exports, and stage attribution."""
import importlib.util
import json
import math
import os
from pathlib import Path
import unittest
import uuid
import weakref
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class IQAEDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('iqae_diagnostics'),
                             'bounded IQAE diagnostics must be available')
        import iqae_diagnostics
        self.diag=iqae_diagnostics
        base=Path(os.environ.get('VRQO_OUTPUT_DIR',ROOT/'outputs'/'storage_iqae_revision_20261109'/'iqae'/'tests'))
        base.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix='iqae_test_',dir=base)
        self.addCleanup(temporary.cleanup)
        self.out=Path(temporary.name)

    def tiny(self):
        from qiskit import QuantumCircuit,QuantumRegister
        status=QuantumRegister(2,'status');g=QuantumRegister(1,'g');work=QuantumRegister(1,'work')
        A=QuantumCircuit(status,g,work)
        A.ry(2*math.asin(math.sqrt(.2)),status[0])
        return A,status,g

    def test_exact_diff_reports_next_float_and_nested_custom_definition(self):
        from qiskit import QuantumCircuit
        a=QuantumCircuit(2,1);b=QuantumCircuit(2,1)
        a.rz(.1,0);b.rz(math.nextafter(.1,math.inf),1)
        ca=QuantumCircuit(1,name='custom');ca.x(0)
        cb=QuantumCircuit(1,name='custom');cb.z(0)
        a.append(ca.to_gate(),[0]);b.append(cb.to_gate(),[0])
        a.measure(0,0);b.measure(1,0);b.global_phase=.01
        report=self.diag.compare_circuits(a,b,max_differences=4)
        self.assertFalse(report['equal'])
        self.assertEqual(report['difference_counts']['params'],1)
        self.assertEqual(report['difference_counts']['qubits'],2)
        self.assertEqual(report['difference_counts']['global_phase'],1)
        self.assertGreaterEqual(report['difference_counts']['definitions'],1)
        self.assertEqual(report['first_differences'][0]['field'],'global_phase')
        self.assertLessEqual(len(report['first_differences']),4)
        self.assertIn('0x1.999999999999a',json.dumps(report))

    def test_custom_definition_diff_identifies_the_first_nested_operation(self):
        from qiskit import QuantumCircuit
        left=QuantumCircuit(1);right=QuantumCircuit(1)
        a=QuantumCircuit(1,name='same');a.x(0)
        b=QuantumCircuit(1,name='same');b.z(0)
        left.append(a.to_gate(),[0]);right.append(b.to_gate(),[0])
        result=self.diag.compare_circuits(left,right,max_differences=1)
        detail=result['first_differences'][0]
        self.assertEqual(detail['field'],'definitions')
        self.assertEqual(detail['nested_first_difference']['field'],'operations')
        self.assertEqual(detail['nested_first_difference']['left'],'x')
        self.assertEqual(detail['nested_first_difference']['right'],'z')

    def test_layout_mapping_difference_is_not_hidden_by_gate_only_hash(self):
        from qiskit import QuantumCircuit,transpile
        q=QuantumCircuit(2);q.x(0)
        left=transpile(q,coupling_map=[[0,1],[1,0]],initial_layout=[0,1],optimization_level=0)
        right=transpile(q,coupling_map=[[0,1],[1,0]],initial_layout=[1,0],optimization_level=0)
        result=self.diag.compare_circuits(left,right)
        self.assertEqual(result['difference_counts']['layout'],1)

    def test_compile_exports_production_measured_circuit_and_no_sampling(self):
        from iqae import OracleSpec
        from qiskit.quantum_info import Statevector
        from runtime_identity import circuit_fingerprint
        A,status,g=self.tiny()
        report=self.diag.run_diagnostic(A_circuit=A,status_qubits=status,g_qubits=g,
            spec=OracleSpec(),powers=[0,1],outdir=self.out,seed=9,mode='compile')
        self.assertEqual(report['status'],'complete')
        self.assertFalse(report['estimation_performed']);self.assertFalse(report['backend_submitted'])
        self.assertIsNone(report['estimate']);self.assertFalse(report['precision_target_met'])
        for row,p in zip(report['powers'],(.2,.968)):
            path=self.out/row['circuits']['compiled_premeasurement']['path']
            qc=self.diag.load_qpy(path)
            self.assertEqual(circuit_fingerprint(qc),row['circuits']['compiled_premeasurement']['fingerprint'])
            got=Statevector.from_instruction(qc).probabilities([report['objective_index']])[1]
            self.assertAlmostEqual(got,p,places=12)
            measured=self.diag.load_qpy(self.out/row['circuits']['compiled_measured']['path'])
            self.assertEqual(measured.count_ops()['measure'],1)
            self.assertEqual(len(qc.data),len(measured.data)-1)
            measured.data.pop()
            self.assertTrue(self.diag.compare_circuits(qc,measured)['equal'])
        for artifact in report['components'].values():
            loaded=self.diag.load_qpy(self.out/artifact['path'],expected_sha256=artifact['sha256'])
            self.assertEqual(circuit_fingerprint(loaded),artifact['fingerprint'])
        self.assertLess((self.out/'diagnostic.json').stat().st_size,100000)

    def test_previous_compiled_power_is_released_before_next_compile(self):
        from iqae import OracleSpec,_PubsToAerObjectiveSampler
        original=_PubsToAerObjectiveSampler._compile_for_aer
        references=[]
        def track(sampler,*args,**kwargs):
            self.assertTrue(all(ref() is None for ref in references))
            compiled=original(sampler,*args,**kwargs)
            references.append(weakref.ref(compiled));return compiled
        A,status,g=self.tiny()
        with patch.object(_PubsToAerObjectiveSampler,'_compile_for_aer',track):
            result=self.diag.run_diagnostic(A_circuit=A,status_qubits=status,g_qubits=g,
                spec=OracleSpec(),powers=[0,1],outdir=self.out,seed=9)
        self.assertEqual(result['status'],'complete',result)

    def test_historical_cloud_identity_is_reported_without_claiming_certification(self):
        report={'seed':2026110803,'case_metadata':{'model_locator':'q_uniform_small','steps':3,
            'profile':{'name':'baseline'}},'components':{'Q':{'fingerprint':'local-Q'}},
            'powers':[{'power':6,'circuits':{'compiled_measured':{'fingerprint':'local-compiled'}}}]}
        result=self.diag.historical_identity_comparison(report)
        self.assertEqual(result['status'],'mismatch_unresolved')
        self.assertFalse(result['rows'][0]['matches'])
        self.assertFalse(result['numerical_equivalence_certified'])
        report['seed']=7
        self.assertEqual(self.diag.historical_identity_comparison(report)['status'],'not_applicable')

    def test_export_does_not_overwrite_and_detects_corruption(self):
        from qiskit import QuantumCircuit
        q=QuantumCircuit(1);q.x(0)
        path=self.out/'circuit.qpy.gz'
        meta=self.diag.export_qpy(q,path)
        self.assertTrue(self.diag.compare_qpy(path,path)['equal'])
        with self.assertRaises(FileExistsError):self.diag.export_qpy(q,path)
        with path.open('ab') as f:f.write(b'corrupted')
        with self.assertRaises(ValueError):self.diag.load_qpy(path,expected_sha256=meta['sha256'])

    def test_wide_control_qpy_roundtrip_preserves_exact_gate_identity(self):
        from qiskit import QuantumCircuit
        from runtime_identity import circuit_fingerprint
        q=QuantumCircuit(55);q.mcx(list(range(54)),54)
        expected=circuit_fingerprint(q)
        path=self.out/'wide_control.qpy.gz'
        artifact=self.diag.export_qpy(q,path)
        restored=self.diag.load_qpy(path,expected_sha256=artifact['sha256'])
        self.assertEqual(circuit_fingerprint(restored),expected)

    def test_compile_budget_stops_before_exporting_oversized_compiled_power(self):
        from iqae import OracleSpec
        A,status,g=self.tiny()
        report=self.diag.run_diagnostic(A_circuit=A,status_qubits=status,g_qubits=g,
            spec=OracleSpec(),powers=[0],outdir=self.out,seed=9,max_compiled_gates=1)
        self.assertEqual(report['status'],'budget_limited')
        self.assertFalse(report['backend_submitted']);self.assertFalse(report['powers'])
        self.assertEqual(report['failed_stage'],'compiled_gate_budget')

    def test_probability_uses_actual_compiled_gates_and_retains_raw_norm(self):
        from iqae import OracleSpec
        A,status,g=self.tiny()
        report=self.diag.run_diagnostic(A_circuit=A,status_qubits=status,g_qubits=g,
            spec=OracleSpec(),powers=[0,1],outdir=self.out,seed=9,mode='probability',
            backend_options={'matrix_product_state_max_bond_dimension':16,
                             'matrix_product_state_truncation_threshold':0.0})
        self.assertEqual(report['status'],'complete',report)
        self.assertTrue(report['backend_submitted']);self.assertIsNone(report['estimate'])
        for row,p in zip(report['powers'],(.2,.968)):
            d=row['probability']
            self.assertAlmostEqual(d['objective_probability'][1],p,places=11)
            self.assertAlmostEqual(d['tensor_norm_before'],1.,places=11)
            self.assertLess(max(d['work_qubit_p1'].values()),1e-12)
            self.assertIn('input_qubit_map',d['backend_metadata'])

    def test_sparse_failure_is_retained_without_pruning_or_renormalizing(self):
        from qiskit import QuantumCircuit
        q=QuantumCircuit(4);q.h(range(4))
        result=self.diag.sparse_probability(q,objective_index=0,max_sparse_states=4)
        self.assertEqual(result['status'],'state_budget_exceeded')
        self.assertIsNone(result['objective_probability'])
        self.assertFalse(result['pruning']);self.assertIsNone(result['wall_deadline_seconds'])

    def test_real_sampler_attributes_fingerprint_failure_before_submission(self):
        from qiskit import QuantumCircuit
        from iqae import _PubsToAerObjectiveSampler
        from ae_runtime import RunBudget
        sampler=_PubsToAerObjectiveSampler(shots=1,objective_qubit_index=0)
        sampler._budget=RunBudget(max_power=0)
        with patch('runtime_identity.circuit_fingerprint',side_effect=MemoryError('fingerprint allocation')):
            with self.assertRaises(MemoryError):sampler.run([(QuantumCircuit(1),)])
        events=sampler._budget.events
        self.assertEqual([e for e in events if e['stage']=='compile_error'][0]['failed_stage'],'fingerprint')
        self.assertTrue(any(e['stage']=='fingerprint_error' and e['resources']['rss_bytes']>0 for e in events))
        self.assertEqual(sampler._budget.summary()['shots_submitted'],0)

    def test_resource_snapshot_distinguishes_rss_from_commit(self):
        result=self.diag.resource_snapshot()
        self.assertGreater(result['rss_bytes'],0)
        if os.name=='nt':
            self.assertGreater(result['process_commit_bytes'],0)
            self.assertGreaterEqual(result['process_peak_commit_bytes'],result['process_commit_bytes'])
            self.assertGreater(result['system_commit_limit_bytes'],result['system_commit_bytes'])


if __name__=='__main__':unittest.main()
