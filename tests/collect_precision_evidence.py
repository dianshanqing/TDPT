"""Reconcile immutable evidence and compile production components without evolution."""
import hashlib
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import load_model
from runtime_identity import circuit_fingerprint, digest as canonical_hash, software_versions, write_json_new
from unitary_circuit import build_unitary_circuit_qiskit, resolve_kernel_profile
from validation import _probe_accuracy_status
from reference_engine import registers
from qiskit import transpile
from qiskit_aer.library import save_probabilities
from iqae import OracleSpec, build_grover_operator_for_spec
from result_storage import read_json,content_digest

OUT = ROOT / 'outputs/precision_1pct_20261106/evidence'
COMPILE = dict(basis_gates=['u','cx'], optimization_level=0, seed_transpiler=917, num_processes=1)


def read(path):
    return read_json(ROOT / path)


def source(path):
    p=ROOT/path
    return {'path':path,**content_digest(p)}


def resource(qc):
    return {'qubits':qc.num_qubits,'gates':len(qc.data),'cx':int(qc.count_ops().get('cx',0)),
            'depth':qc.depth(),'fingerprint':circuit_fingerprint(qc)}


def components():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for tau in (14,19,23):
        cfg=load_model('compact_n8_w0002_tau'+str(tau))
        row={'tau':tau,'steps':15,'config':asdict(cfg),'config_hash':canonical_hash(asdict(cfg)),
             'compile':COMPILE,'simulation_executed':False,'profiles':{}}
        for profile in ('baseline','compressed'):
            tick=time.perf_counter()
            A,_=build_unitary_circuit_qiskit(cfg,15,kernel_profile=profile)
            compiled=transpile(A,**COMPILE)
            observed=compiled.copy();regs=registers(compiled)
            observed.save_probabilities(regs['status'],label='status_probability')
            for i,bit in enumerate(regs['work']):observed.save_probabilities([bit],label=f'work_{i}')
            entry={'internal_profile':asdict(resolve_kernel_profile(kernel_profile=profile)),
                   'original_A':resource(A),'compiled_A':resource(compiled),
                   'observation':resource(observed),'registers':regs}
            if profile=='baseline':
                q,objective,_=build_grover_operator_for_spec(A_circuit=A,
                    status_qubits=next(r for r in A.qregs if r.name=='status'),g_qubits=next(r for r in A.qregs if r.name=='g'),
                    spec=OracleSpec(good='detect'))
                entry['objective_qubits']=objective
                entry['Atilde']=resource(transpile(q.state_preparation,**COMPILE))
                entry['Atilde_inverse']=resource(transpile(q.state_preparation.inverse(),**COMPILE))
                entry['S0']=resource(transpile(q.zero_reflection,**COMPILE))
                entry['Sf']=resource(transpile(q.oracle,**COMPILE))
                entry['resource_scope']='Actual independent component compilations; no expanded Q^k A and no state evolution. Sum of component depths is a compositional upper bound, not optimized whole-circuit depth.'
            entry['build_compile_fingerprint_seconds']=time.perf_counter()-tick
            row['profiles'][profile]=entry
            print('components',tau,profile,entry['compiled_A']['gates'],entry['build_compile_fingerprint_seconds'],flush=True)
        write_json_new(OUT/f'components_tau{tau}.json',row);rows.append(row)
    write_json_new(OUT/'production_components.json',{'rows':rows,'versions':software_versions(),
        'production_public_name':'baseline','production_internal_implementation':'baseline',
        'reason':'Same qubit count; fewer compiled gates, CX and depth than historical compressed. No MPS speed claim.',
        'historical_profile_semantics_preserved':True})
    return rows


def reconcile(rows):
    audit_path='outputs/revision_screening/20261102_tau23_cmd_01/audit_tau23.json'
    audit=read(audit_path)
    records=[]
    for tau,bond,path in [(14,64,'outputs/revision_screening/20261030_compact_deep_01/quantum/scatter_tau14/mps_probability_probe.json'),
                          (23,256,'cloud/cloud_tau23_b256_01/mps_probability_probe.json')]:
        report=read(path);r=next(x for x in report['rows'] if x['bond']==bond and x['truncation_threshold']==1e-16)
        records.append({'tau':tau,'bond':bond,'source':source(path),'source_pointer':'/rows/'+str(report['rows'].index(r)),
            'evidence_kind':'original_result','execution_status':r['status'],
            'probabilities':r['probabilities'],'reference':r['reference'],
            'work_upper_bound':r['work_any_nonzero_union_upper_bound'],
            'fingerprints':{k:r[k] for k in ['original_A_fingerprint','compiled_A_fingerprint','diagnostic_fingerprint']},
            'config':r['config'],'actual_profile':r['identity']['actual_profile'],'compile':r['identity']['compile'],
            'backend_options':r['backend_options'],'identity_hash':r['identity']['identity_hash'],
            'peak_process_memory_mb':r['peak_process_memory_mb'],
            'historical_judgement':{k:r.get(k) for k in ['passes_1_percent','passes_5_percent','passed_absolute_tolerance']}})
    for tau in (19,23):
        r=audit['runs'][str(tau)]
        records.append({'tau':tau,'bond':128,'source':source(audit_path),'source_pointer':'/runs/'+str(tau),
            'evidence_kind':'retained_audit_of_original_result','original_source_path':r['source_file'],
            'original_source_currently_available':Path(r['source_file']).is_file(),
            'original_source_sha256':next((v['sha256'] for k,v in audit['raw_manifest'].items() if k.endswith(Path(r['source_file']).name)),None),
            'execution_status':r['execution_status'],'probabilities':r['probabilities'],'reference':r['reference'],
            'work_upper_bound':r['work_upper_bound'],'fingerprints':r['fingerprints'],'config':r['config'],
            'actual_profile':r['actual_profile'],'compile':r['compile'],'backend_options':r['backend_options'],
            'identity_hash':r['identity_hash'],'historical_judgement':{k:r[k] for k in ['passes_formal_1pct','passes_pilot_5pct','strict_status']}})
    classical_path='outputs/revision_screening/20261030_compact_deep_01/precision_classical.json'
    classical=read(classical_path)
    screen_path='outputs/revision_screening/20261030_compact_deep_01/compact_screen.json'
    screen=read(screen_path)
    for record in records:
        build=next(r for r in rows if r['tau']==record['tau'])
        record['current_assessment']=_probe_accuracy_status(record['probabilities'],record['reference'],record['work_upper_bound'])
        record['relative_D_error']=abs(record['probabilities'][1]-record['reference'][1])/record['reference'][1]
        record['reference_identity']={'config_hash':canonical_hash(record['config']),'steps':15,
            'target':'D total for the identical fixed discrete model','high_precision_source':source(classical_path),
            'matching_rows':[r for r in classical['rows'] if r['id']==f'compact_n8_w0002_tau{record["tau"]}' and r['steps']==15]}
        record['classical_tail_evidence']={'source':source(screen_path),
            'matching_rows':[r for r in screen['rows'] if r['id']==f'compact_n8_w0002_tau{record["tau"]}']}
        record['classical_grid_evidence']={'source':source(screen_path),
            'matching_rows':[r for r in screen['refinements'] if r['omega']==.002 and r['tau']==record['tau'] and r['base_cells']==8],
            'scope':'Existing grid differences, not a proof of continuum convergence; separate from fixed-model 1% response.'}
        record['model_matches_current']=canonical_hash(record['config'])==canonical_hash(build['config'])
        old=build['profiles']['compressed'];new=build['profiles']['baseline']
        checks={'original_A_fingerprint':'original_A','compiled_A_fingerprint':'compiled_A','diagnostic_fingerprint':'observation'}
        record['matches_rebuilt_historical_implementation']={k:record['fingerprints'][k]==old[v]['fingerprint'] for k,v in checks.items()}
        record['matches_production_implementation']={k:record['fingerprints'][k]==new[v]['fingerprint'] for k,v in checks.items()}
        record['reuse_for_historical_implementation']=all(record['matches_rebuilt_historical_implementation'].values()) and record['model_matches_current']
        record['reuse_for_production_validation']=all(record['matches_production_implementation'].values()) and record['model_matches_current']
        record['reuse_note']='Historical execution and response evidence retained with original judgement. A/profile/compiled/observation mismatch prevents certification of production baseline; no rerun for reporting changes alone.'
    extra_sources=['outputs/revision_screening/20261030_compact_deep_01/compact_screen.json',classical_path,
                   'outputs/revision_screening/20261102_cloud_review_01/reference_review.json']
    report={'rows':records,'status_order':['survive','detect','absorb','boundary'],
        'classical_evidence_reused':[source(p) for p in extra_sources if (ROOT/p).is_file()],
        'no_full_MPS_evolution':True,'no_new_grid_quantum_simulation':True,
        'cloud_branch':'B','first_required_cloud_case':{'tau':23,'steps':15,'profile':'baseline','bond':256,
            'backend_options':next(r for r in records if r['bond']==256)['backend_options'],
            'cpu_budget':50,'max_cpu_budget':60,'memory_mib':8192,'wall_clock_deadline':None,
            'memory_rationale':'Historical compressed worker peak 933.74 MiB; 8192 MiB retained with headroom, production baseline memory unmeasured.'},
        'unresolved_numerical_control':{'proposed_only':True,'automatic_bond_increase':False,
            'minimal_control':'If new baseline still has material normalization/work anomalies: same baseline, bond256, trunc1e-16, seed and marginal-only observation; change only mps_lapack false to true after user review.',
            'discriminator':'Changes larger than target D budget or normalization residual implicate SVD implementation sensitivity; stable errors do not prove truncation harmless. Tensor-save readout changes are separately identified and cannot normalize/correct probabilities.'}}
    write_json_new(OUT/'historical_reuse_audit.json',report)
    print('audit',[(r['tau'],r['bond'],r['relative_D_error'],r['reuse_for_historical_implementation']) for r in records],flush=True)


def ae_components():
    """Compile the same independent gates using the real Aer IQAE basis."""
    options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=0,seed_transpiler=20261030)
    rows=[]
    for tau in (14,19,23):
        cfg=load_model('compact_n8_w0002_tau'+str(tau))
        A,_=build_unitary_circuit_qiskit(cfg,15,kernel_profile='baseline')
        q,objective,_=build_grover_operator_for_spec(A_circuit=A,
            status_qubits=next(r for r in A.qregs if r.name=='status'),
            g_qubits=next(r for r in A.qregs if r.name=='g'),spec=OracleSpec(good='detect'))
        row={'tau':tau,'config_hash':canonical_hash(asdict(cfg)),'profile':'baseline','steps':15,
             'compile':options,'objective_qubits':objective,'simulation_executed':False}
        for key,circuit in [('Atilde',q.state_preparation),('Atilde_inverse',q.state_preparation.inverse()),
                            ('S0',q.zero_reflection),('Sf',q.oracle)]:
            row[key]=resource(transpile(circuit,**options))
        rows.append(row)
        print('ae_components',tau,{k:row[k]['gates'] for k in ('Atilde','Atilde_inverse','S0','Sf')},flush=True)
    write_json_new(OUT/'production_ae_components.json',{'rows':rows,'versions':software_versions(),
        'compile_scope':'Same safe basis and seed as iqae._PubsToAerObjectiveSampler._compile_for_aer; independently compiled components. No whole expanded Q^k A circuit, no evolution.',
        'depth_scope':'Sum of independent component depths is a compositional upper bound, not measured whole expanded circuit depth.',
        'measurement_cost':'One final objective measurement per shot; separately add one measurement to gates/depth if included.'})


if __name__=='__main__':
    if '--ae-components' in sys.argv:ae_components()
    else:reconcile(read('outputs/precision_1pct_20261106/evidence/production_components.json')['rows']
                   if '--reuse-components' in sys.argv else components())
