"""Actual gate tests. Run serially under supervised, one profile per job."""
import json,math,os,sys,time,unittest
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from reference_engine import evolve,reference
from config import default_config
from unitary_circuit import build_unitary_circuit_qiskit
from iqae import OracleSpec,build_grover_operator_for_spec


class ActualAEContractTests(unittest.TestCase):
    def check_profile(self,profile):
        from runtime_identity import circuit_fingerprint,source_manifest,software_versions
        started=time.monotonic();cfg=default_config();cfg=replace(cfg,geom=replace(cfg.geom,detector_x=2))
        A,_=build_unitary_circuit_qiskit(cfg,3,kernel_profile=profile)
        regs={r.name:r for r in A.qregs};state,_=evolve(A,timeout=None,max_states=300000)
        back,_=evolve(A.inverse(),state,timeout=None,max_states=300000)
        inverse_error=abs(back.get(0,0)-1);tail=math.fsum(abs(v)**2 for k,v in back.items() if k)
        self.assertLess(inverse_error,2e-11);self.assertLess(tail,2e-20)
        p_detect=reference(cfg,3)['rows'][-1]['probabilities'][1];checks=[]
        for label,spec,p in [('detect',OracleSpec(good='detect'),p_detect),('g0',OracleSpec(good='detect',target_g=0),p_detect),
            ('g1_zero',OracleSpec(good='detect',target_g=1),0.),('rare',OracleSpec(good='rare_detect',rare_eta=1e-4),p_detect*1e-4)]:
            Q,obj,_=build_grover_operator_for_spec(A_circuit=A,status_qubits=regs['status'],g_qubits=regs['g'],spec=spec)
            self.assertEqual(list(Q.reflection_qubits),list(range(Q.num_qubits)))
            psi,_=evolve(Q.state_preparation,timeout=None,max_states=300000);rows=[]
            for k in range(3):
                got=math.fsum(abs(v)**2 for n,v in psi.items() if (n>>obj[0])&1)
                want=math.sin((2*k+1)*math.asin(math.sqrt(p)))**2
                self.assertLess(abs(got-want),2e-11)
                rows.append({'k':k,'observed':got,'expected_from_independent_DP':want,'absolute_error':abs(got-want)})
                if k<2:psi,_=evolve(Q,psi,timeout=None,max_states=300000)
            checks.append({'target':label,'p':p,'rare_eta':spec.rare_eta,'p_rare':p if spec.rare_eta is not None else None,
                'state_preparation_fingerprint':circuit_fingerprint(Q.state_preparation),'grover_fingerprint':circuit_fingerprint(Q),
                'objective_indices':obj,'reflection_qubits':list(Q.reflection_qubits),'all_registers_reflected':True,'iterations':rows})
        out=os.environ.get('VRQO_OUTPUT_DIR')
        if out:
            from runtime_identity import write_json_new
            write_json_new(Path(out)/f'quantum_ae_{profile}.json',{'profile':profile,'steps':3,'config':asdict(cfg),
                'A_fingerprint':circuit_fingerprint(A),'register_map':{r.name:[A.find_bit(q).index for q in r] for r in A.qregs},
                'sources':source_manifest(),'software_versions':software_versions(),'inverse_amplitude_error':inverse_error,
                'inverse_tail_probability':tail,'checks':checks,'elapsed_seconds':time.monotonic()-started,'probability_tolerance':2e-11,
                'evolution_timeout_seconds':None,'evolution_max_states':300000})
    def test_baseline(self):self.check_profile('baseline')
    def test_compressed(self):self.check_profile('compressed')
    def test_full_history(self):self.check_profile('full_history')

if __name__=='__main__':unittest.main()
