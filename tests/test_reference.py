"""Independent checks of the portable test engine, before trusting transport tests."""
import json
import math
import os
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate,RZGate
from qiskit.quantum_info import Statevector
from reference_engine import evolve,reference,config_with_probs
from config import default_config,paper_scale_config
from unitary_circuit import _U6


class ReferenceTests(unittest.TestCase):
    def test_sparse_against_independent_dense_complex_engine(self):
        for seed in (1,2,3):
            rng=np.random.default_rng(seed);qc=QuantumCircuit(7);qc.global_phase=.123*seed
            for b in range(7):qc.ry(float(rng.uniform(-2,2)),b);qc.rz(float(rng.uniform(-2,2)),b)
            qc.append(RYGate(.47*seed).control(4,ctrl_state=5,annotated=False),[0,2,4,6,1])
            qc.append(RZGate(.38).control(3,ctrl_state=2,annotated=False),[5,1,3,0])
            qc.append(_U6.control(1),[2,6,3,0]);qc.h(2);qc.mcx([0,2,5],4);qc.swap(1,5)
            for circuit in (qc,qc.inverse()):
                st,_=evolve(circuit,timeout=None)
                self.assertLess(float(np.max(abs(Statevector.from_instruction(circuit).data-np.array([st.get(i,0) for i in range(128)])))),2e-12)

    def test_nonunitary_rejected(self):
        for gate in ('reset','measure'):
            qc=QuantumCircuit(1,1)
            if gate=='reset':qc.reset(0)
            else:qc.measure(0,0)
            with self.assertRaises(ValueError):evolve(qc,timeout=None)

    def test_classical_dp_against_closed_form_absorber(self):
        cfg=config_with_probs(default_config(),.2,1)
        for row in reference(cfg,15)['rows']:
            t=row['steps'];live=.8**min(t,11)
            expected=[live if t<12 else 0,live if t>=12 else 0,1-live,0]
            self.assertLess(max(abs(a-b) for a,b in zip(row['probabilities'],expected)),2e-12)

    def test_default_classical_reference_t1_to_t15(self):
        result=reference(default_config(),15)
        for row in result['rows']:self.assertLess(row['norm_error'],2e-12)
        expected={12:[.10935921120416062,.43464278993253214,.08287687006630295,.37312112879700404],15:[.03587956701409075,.4375588180978207,.08538736905148205,.44117424583660636]}
        for t,values in expected.items():
            self.assertLess(max(abs(a-b) for a,b in zip(result['rows'][t-1]['probabilities'],values)),2e-12)
        if 'VRQO_OUTPUT_DIR' in os.environ:
            (Path(os.environ['VRQO_OUTPUT_DIR'])/'independent_dp.json').write_text(json.dumps(result,indent=2),encoding='utf8')

    def test_existing_mc_four_group_downscatter_is_preserved(self):
        from mc import run_mc
        cfg=config_with_probs(paper_scale_config(),1,0,1)
        cfg=replace(cfg,geom=replace(cfg.geom,detector_x=2))
        result=run_mc(cfg,2,256,seed=20261026)
        self.assertEqual(sum(result.counts.values()),256)
        self.assertGreater(result.counts['detect'],0)
        self.assertEqual(float(result.energy_hist_detect[1]),1.)  # MC exposes a conditional normalized histogram.
        self.assertEqual(int(result.energy_hist_detect[[0,2,3]].sum()),0)

if __name__=='__main__':unittest.main()
