"""Minimal regressions for the declared solid and layered material geometry."""
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from config import MaterialLayer, default_config, validate_config
from geometry import material_at
from models import load_model, model_from_dict, config_dict


class TestRevisionModels(unittest.TestCase):
    def test_explicit_cloud_sparse_skip_never_evolves_or_fills_zero(self):
        from validation import _sparse_crosscheck
        with patch('reference_engine.evolve') as evolve:
            report = _sparse_crosscheck(None, [1,0,0,0], enabled=False)
        evolve.assert_not_called()
        self.assertIsNone(report['sparse_actual'])
        self.assertIsNone(report['sparse_runtime'])
        self.assertFalse(report['sparse_crosscheck'])
        self.assertIsNone(report['sparse_passed'])
        self.assertIn('explicit', report['sparse_skip_reason'].lower())

    def test_sparse_crosscheck_defaults_on_and_requires_boolean(self):
        from validation import _sparse_crosscheck
        with patch('reference_engine.evolve', return_value=({0:1}, {'seconds':1})) as evolve:
            with patch('reference_engine.observe', return_value={'norm':1,'probabilities':[1,0,0,0],'work_nonzero':0}):
                report = _sparse_crosscheck('circuit', [1,0,0,0])
        evolve.assert_called_once_with('circuit', max_states=60000)
        self.assertTrue(report['sparse_crosscheck'])
        self.assertTrue(report['sparse_passed'])
        with self.assertRaises(ValueError):
            _sparse_crosscheck(None, [1,0,0,0], enabled='false')

    def test_invalid_suite_steps_and_empty_probe_settings_rejected(self):
        from validation import _validated_cases, _probe_settings, _cpu
        for value in (2.9, True, 0, -1, '2', 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validated_cases({'cases':[{'steps':value}]})
        for key in ('bond_dimensions','truncation_thresholds'):
            spec={'bond_dimensions':[16], 'truncation_thresholds':[1e-12], 'shots':256}
            spec[key]=[]
            with self.assertRaises(ValueError):
                _probe_settings(spec)
        for value in (True, 1.5):
            with self.assertRaises(ValueError):
                _cpu(value)

    def test_sparse_incorrect_status_or_work_fails_explicitly(self):
        from validation import _sparse_crosscheck
        for observed in ({'norm':1,'probabilities':[.9,.1,0,0],'work_nonzero':0},
                         {'norm':1,'probabilities':[1,0,0,0],'work_nonzero':1e-5}):
            with patch('reference_engine.evolve', return_value=({0:1}, {})):
                with patch('reference_engine.observe', return_value=observed):
                    report=_sparse_crosscheck('circuit',[1,0,0,0])
            self.assertFalse(report['sparse_passed'])

    def test_load_roundtrip_and_reject_misspelled_field(self):
        cfg = load_model('q_layered_small')
        self.assertEqual(model_from_dict(config_dict(cfg)), cfg)
        with self.assertRaises(ValueError):
            model_from_dict(dict(config_dict(cfg), layres=[]))

    def test_nonbinary_grid_layer_controls_and_uncomputation(self):
        from qiskit import QuantumCircuit, QuantumRegister
        from tests.test_geometry_config import permute_basis
        from unitary_circuit import compile_material_predicates, uncompile_material_predicates
        for name in ('uniform', 'layered'):
            cfg = load_model(f'q_{name}_small')
            x,y,z = QuantumRegister(3,'x'),QuantumRegister(2,'y'),QuantumRegister(2,'z')
            w = QuantumRegister(10,'work')
            forward=QuantumCircuit(x,y,z,w); backward=QuantumCircuit(x,y,z,w)
            a,b=compile_material_predicates(forward,x,y,z,list(w),cfg)
            uncompile_material_predicates(backward,x,y,z,list(w),cfg)
            for ix in range(cfg.geom.nx):
                for iy in range(cfg.geom.ny):
                    for iz in range(cfg.geom.nz):
                        initial=ix | (iy<<3) | (iz<<5)
                        actual=permute_basis(forward,initial)
                        self.assertEqual((actual>>forward.find_bit(a).index)&1, int(1<=ix<=2))
                        self.assertEqual((actual>>forward.find_bit(b).index)&1, int(name=='layered' and ix==2))
                        self.assertEqual(permute_basis(backward,actual),initial)

    def test_new_models_actual_joint_and_zero_work(self):
        from reference_solver import reference
        from reference_engine import evolve, marginal, observe, build_traced
        from runtime_identity import circuit_fingerprint
        for name in ('legacy','uniform','layered'):
            cfg=load_model(f'q_{name}_small')
            for profile in ('baseline','compressed','full_history'):
                for steps in (3,5):
                    with self.subTest(model=name,profile=profile,steps=steps):
                        circuit,_,points=build_traced(cfg,steps,profile)
                        state,record=evolve(circuit,max_states=60000,checkpoints=points,observer=observe)
                        actual=marginal(circuit,state,['status','x','y','z','dir','g'])
                        expected=reference(cfg,steps,retain_joint=True)
                        joint=expected['joint']
                        error=max(abs(actual.get(k,0)-joint.get(k,0)) for k in actual.keys()|joint.keys())
                        self.assertLess(error,2e-11)
                        self.assertEqual(len(record['checkpoints']),steps)
                        checkpoint_errors=[]
                        for row,want in zip(record['checkpoints'],expected['rows']):
                            seen=row['observables']
                            self.assertLess(seen['work_nonzero'],2e-11)
                            self.assertLess(abs(seen['norm']-1),2e-11)
                            difference=max(abs(a-b) for a,b in zip(seen['probabilities'],want['probabilities']))
                            self.assertLess(difference,2e-11)
                            checkpoint_errors.append(difference)
                        evidence={'model':name,'profile':profile,'steps':steps,
                            'A_fingerprint':circuit_fingerprint(circuit),'max_joint_error':error,
                            'max_step_status_error':max(checkpoint_errors),
                            'max_step_work_nonzero':max(r['observables']['work_nonzero'] for r in record['checkpoints']),
                            'physical_step_checkpoints':steps,'boundary':'full physical step after uncomputation'}
                        if os.environ.get('VRQO_OUTPUT_DIR'):
                            with (Path(os.environ['VRQO_OUTPUT_DIR'])/'model_science.jsonl').open('a',encoding='utf8') as stream:
                                stream.write(json.dumps(evidence)+'\n')

    def test_solid_has_no_duct_and_layer_order_is_explicit(self):
        base = default_config()
        solid = replace(base, model='uniform_slab', layers=(MaterialLayer(8, 9, 'tungsten'),))
        validate_config(solid)
        self.assertEqual(material_at(solid, 8, 3, 3), 'tungsten')
        layered = replace(base, model='layered_slab', layers=(MaterialLayer(8, 8, 'air'), MaterialLayer(9, 9, 'tungsten')))
        validate_config(layered)
        self.assertEqual(material_at(layered, 8, 2, 2), 'air')
        self.assertEqual(material_at(layered, 9, 3, 3), 'tungsten')
        self.assertEqual(material_at(layered, 7, 3, 3), 'concrete')

    def test_gaps_overlaps_unknown_material_and_bad_h_rejected(self):
        base = default_config()
        for layers in ((MaterialLayer(8, 8, 'air'), MaterialLayer(8, 9, 'tungsten')),
                       (MaterialLayer(8, 8, 'air'),),
                       (MaterialLayer(8, 8, 'vacuum'), MaterialLayer(9, 9, 'tungsten'))):
            with self.assertRaises(ValueError):
                validate_config(replace(base, model='layered_slab', layers=layers))
        for h in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                validate_config(replace(base, h=h))


if __name__ == '__main__':
    unittest.main()
