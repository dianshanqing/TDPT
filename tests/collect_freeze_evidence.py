"""Finite Paper1 closeout evidence; no production edits or main MPS evolution.

Run individual stages with the existing QC Python. Each output is exclusive:
an unsuccessful attempt remains evidence and is never silently overwritten.
"""
from pathlib import Path
import sys
import os
import json
import math
import hashlib
import time
from dataclasses import replace, asdict
from decimal import Decimal, localcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/baseline_freeze_20261107'
from runtime_identity import normalize, source_manifest, check_identity, circuit_fingerprint, software_versions
from models import load_model
from reference_solver import reference
from reference_engine import registers, evolve, observe, marginal, getval, flattened, apply_instruction, build_traced
from unitary_circuit import build_unitary_circuit_qiskit

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(normalize(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def audit():
    from validation import _probe_accuracy_status
    old = read('outputs/precision_1pct_20261106/evidence/production_components.json')['rows']
    hp = read('outputs/revision_screening/20261030_compact_deep_01/precision_classical.json')['rows']
    manifest = {}; rows = []; current = source_manifest(); versions = software_versions()
    for tau, bond in ((14,64),(19,128),(23,256)):
        folder = ROOT / f'outputs/direct_response/baseline_tau{tau}_T15_bond{bond}_01'
        for path in sorted(folder.rglob('*')):
            if path.is_file():manifest[path.relative_to(ROOT).as_posix()]={'bytes':path.stat().st_size,'sha256':sha(path)}
        aggregate = read(folder / 'mps_probability_probe.json'); agg = aggregate['rows'][0]
        rawpath = folder / f'n8_w0002_tau{tau}_T15_bond{bond}_trunc1e-16.json'
        raw = read(rawpath); summary = read(folder / 'summary/result.json'); summ = summary['rows'][0]
        sup = read(folder/'supervision_result.json'); cfg = load_model(f'compact_n8_w0002_tau{tau}')
        identity = raw['identity']; check_identity(identity)
        prior = next(x for x in old if x['tau']==tau)['profiles']['baseline']
        classical = reference(cfg,15); ref = classical['rows'][-1]['probabilities']
        hi = next(x for x in hp if x['id']==f'compact_n8_w0002_tau{tau}')['runs']['fixed_float_80']
        actual = raw['probabilities']; work = raw['work_bit_probabilities']
        errors = [abs(a-b) for a,b in zip(actual,ref)]
        with localcontext() as ctx:
            ctx.prec=85
            relative80=abs(Decimal.from_float(actual[1])-Decimal(hi['p_D']))/Decimal(hi['p_D'])
        computed = _probe_accuracy_status(actual,ref,math.fsum(work),work_bit_probabilities=work)
        keys = ('probabilities','reference','work_bit_probabilities','register_map','case','config','identity',
                'original_A_fingerprint','compiled_A_fingerprint','diagnostic_fingerprint','backend_options',
                'strict_status','numerical_validity_status','scientific_certification_passed')
        checks = {f'raw_aggregate_{k}':raw[k]==agg[k] for k in keys}
        checks.update({
            '16_return_files':sum(k.startswith(folder.relative_to(ROOT).as_posix()+'/') for k in manifest)==16,
            'summary_source_sha256':summary['source_sha256']==sha(folder/'mps_probability_probe.json'),
            'summary_D':summ['p_D']==actual[1],
            'summary_labels':all(summ[k]==raw[k] for k in ('strict_status','rare_accuracy_status','numerical_validity_status')),
            'config_matches_current_preset':normalize(asdict(cfg))==raw['config'],
            'T15_baseline':raw['case']['steps']==15 and raw['case']['profile']=='baseline',
            'model_reference_agreement':max(abs(a-b) for a,b in zip(ref,raw['reference']))<2e-15,
            'identity_registers':identity['measurement']['register_map']==raw['register_map']==prior['registers'],
            'original_A':raw['original_A_fingerprint']==prior['original_A']['fingerprint'],
            'compiled_A':raw['compiled_A_fingerprint']==prior['compiled_A']['fingerprint'],
            'observation':raw['diagnostic_fingerprint']==prior['observation']['fingerprint'],
            'compile_settings':identity['compile']==next(x for x in old if x['tau']==tau)['compile'],
            'process_completed':sup['returncode']==0 and sup['termination_reason']=='completed' and sup['all_processes_reaped'],
            'no_deadline':sup['effective_deadline_monotonic'] is None,
            'independent_reference_only':raw['sparse_crosscheck'] is False and raw['compiled_sparse'] is None,
            'counts_disabled':aggregate['counts_check'] is False,
            'pinned_versions':all(v==versions[k] for k,v in identity['versions'].items() if k!='platform'),
            'source_complete':set(identity['sources'])==set(current),
        })
        logref=agg['metadata_log_references']['MPS_log_data']
        checks['raw_log_file_hash']=sha(rawpath)==logref['file_sha256']
        checks['MPS_log_hash']=hashlib.sha256(raw['metadata']['MPS_log_data'].encode()).hexdigest()==logref['log_sha256']
        meta=raw['metadata']; bit=raw['register_map']['work'][4]; active=dict(meta['input_qubit_map'])[bit]
        changes=[k for k,v in current.items() if identity['sources'].get(k)!=v]
        for key in ('rare_accuracy_status','strict_status','numerical_validity_status','work_accuracy_status'):
            checks['recomputed_'+key]=computed[key]==raw[key]
        row={'tau':tau,'bond':bond,'source':rawpath.relative_to(ROOT).as_posix(),
             'checks':checks,'all_integrity_checks_passed':all(checks.values()),
             'original_labels':{k:raw[k] for k in ('status','execution_status','rare_accuracy_status','strict_status','work_accuracy_status','numerical_validity_status','scientific_certification_passed')},
             'probabilities':actual,'reference':ref,'reference_80_digits':hi,
             'reference_recheck':{'seconds':classical.get('elapsed_seconds'),'max_absolute_difference':max(abs(a-b) for a,b in zip(ref,raw['reference']))},
             'absolute_errors':errors,'D_relative_error':errors[1]/ref[1],
             'D_relative_error_percent':100*errors[1]/ref[1],
             'D_relative_error_against_80_digit_fixed_model':str(relative80),
             'raw_probability_sum':math.fsum(actual),'normalization_residual':abs(math.fsum(actual)-1),
             'work_bit_probabilities':work,'work_union_upper_bound':math.fsum(work),
             'work4_mapping':{'original_qubit':bit,'active_qubit':active,'save_label':'work_4','extraction':'data[work_4][1]; original-index save instruction, Aer remaps internally'},
             'register_map':raw['register_map'],'backend_options':raw['backend_options'],
             'blas_threads':raw['blas_threads'],'versions':identity['versions'],
             'actual_source_hashes':identity['sources'],'current_source_differences':changes,
             'compile':identity['compile'],
             'fingerprints':{k:raw[k] for k in ('original_A_fingerprint','compiled_A_fingerprint','diagnostic_fingerprint')},
             'supervision':sup,'recomputed_status':computed,
             'target_impact':{'direct_D_empirical_comparison_passed':errors[1]/ref[1]<=.01,
               'norm_residual_is_D_error_bound':False,'work_union_is_D_error_bound':False,
               'all_distribution_certified':False,'IQAE_certified':False,
               'interpretation':'Independent same-model D comparison supports this observable at these settings. Mass residual and work leakage do not locate missing probability or bound amplified-circuit error. Raw full-distribution diagnostics are retained.'}}
        rows.append(row)
        print(json.dumps({'tau':tau,'all_integrity_checks':all(checks.values()),'failed':[k for k,v in checks.items() if not v],
              'D_relative_percent':row['D_relative_error_percent'],'changed_sources':changes,'mapping':row['work4_mapping']},ensure_ascii=False),flush=True)
    save('direct_response_audit.json',{'rows':rows,'original_file_manifest':manifest,'current_sources':current,
        'no_main_MPS_evolution':True,'probabilities_modified':False,'reference_method':'independent reference_solver float64; retained fixed_float_80 decimal recurrence',
        'all_integrity_checks_passed':all(r['all_integrity_checks_passed'] for r in rows)})

def mps(qc, bond=128, labels=None):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from numerical_runtime import configure_blas_threads
    from validation import _probe_backend_options
    compiled=transpile(qc,basis_gates=['u','cx'],optimization_level=0,seed_transpiler=917,num_processes=1)
    obs=compiled.copy(); regs=registers(qc)
    if labels is None:labels={'status_probability':regs['status'],**{f'work_{i}':[b] for i,b in enumerate(regs['work'])}}
    for name,bits in labels.items():obs.save_probabilities(bits,label=name)
    options=_probe_backend_options(1,1,1,{'precision':'double','zero_threshold':0.,'chop_threshold':0.,'mps_lapack':False},bond,1e-16)
    sim=AerSimulator(**options); configure_blas_threads(1)
    begin=time.perf_counter();cpu=time.process_time();result=sim.run(obs,shots=1).result()
    if not result.success:raise RuntimeError(result.status)
    return {'data':{k:v.tolist() for k,v in result.data().items()},'metadata':{k:v for k,v in result.results[0].metadata.items() if k!='MPS_log_data'},
            'seconds':time.perf_counter()-begin,'cpu_seconds':time.process_time()-cpu,'options':options,
            'A_fingerprint':circuit_fingerprint(qc),'compiled_fingerprint':circuit_fingerprint(compiled),
            'observation_fingerprint':circuit_fingerprint(obs),'compiled_gates':len(compiled.data),'compiled_depth':compiled.depth()}

def work_diagnostic():
    from qiskit import QuantumCircuit, QuantumRegister
    import unitary_circuit as uc
    from tests.test_geometry_config import permute_basis
    cfg=load_model('compact_n8_w0002_tau19')
    x,y,z=QuantumRegister(5,'x'),QuantumRegister(4,'y'),QuantumRegister(4,'z')
    w=QuantumRegister(17,'work'); h=QuantumRegister(4,'predicate_history')
    forward=QuantumCircuit(x,y,z,w,h)
    a,b=uc.compile_material_predicates(forward,x,y,z,list(w)[2:12],cfg)
    forward.cx(a,h[0]);forward.cx(b,h[1]);uc.uncompile_material_predicates(forward,x,y,z,list(w)[2:12],cfg)
    uc.compile_boundary_predicate(forward,x,y,z,w[0],list(w)[2:8],list(w)[8:12],cfg)
    forward.cx(w[0],h[2]);uc.uncompile_boundary_predicate(forward,x,y,z,w[0],list(w)[2:8],list(w)[8:12],cfg)
    flag=uc.compile_roi_predicate(forward,x,y,z,list(w)[2:11],cfg)
    forward.cx(flag,h[3]);uc.uncompile_roi_predicate(forward,x,y,z,list(w)[2:11],cfg)
    regs=registers(forward); failures=[]
    for ix in range(17):
        for iy in range(9):
            for iz in range(9):
                initial=ix|(iy<<5)|(iz<<9);actual=permute_basis(forward,initial)
                history=int(4<=ix<=11)|(int(ix in (0,16) or iy in (0,8) or iz in (0,8))<<2)|(int(ix==14 and 3<=iy<=5 and 3<=iz<=5)<<3)
                expected=initial|(history<<regs['predicate_history'][0])
                if actual!=expected:failures.append([ix,iy,iz,actual,expected])
    points=[(3,4,4),(4,4,4),(11,4,4),(12,4,4),(4,0,4),(4,8,4),(0,4,4),(16,4,4),
            (14,3,3),(14,5,5),(14,2,4),(4,4,0),(4,4,8)]
    comparisons=[]
    for ix,iy,iz in points:
        qc=QuantumCircuit(*forward.qregs)
        for val,reg in ((ix,x),(iy,y),(iz,z)):
            for bit in range(len(reg)):
                if (val>>bit)&1:qc.x(reg[bit])
        qc.compose(forward,inplace=True)
        state,info=evolve(qc,max_states=1000)
        labels={'history':regs['predicate_history'],**{f'work_{i}':[b] for i,b in enumerate(regs['work'])}}
        numeric=mps(qc,labels=labels)
        expect=[0.]*16
        for key,amp in state.items():expect[getval(key,regs['predicate_history'])]+=abs(amp)**2
        comparisons.append({'position':[ix,iy,iz],'independent_complex':info,'history_expected':expect,
            'MPS':numeric,'history_max_absolute_error':max(abs(a-b) for a,b in zip(expect,numeric['data']['history'])),
            'work_union_upper_bound':math.fsum(numeric['data'][f'work_{i}'][1] for i in range(17))})
    qc,_,checkpoints=build_traced(cfg,4,'baseline')
    state,info=evolve(qc,max_states=60000,checkpoints=checkpoints,observer=observe)
    ref=reference(cfg,4,retain_joint=True); joint=marginal(qc,state,['status','x','y','z','dir','g'])
    joint_error=max(abs(joint.get(k,0)-ref['joint'].get(k,0)) for k in set(joint)|set(ref['joint']))
    numeric=mps(qc);p=numeric['data']['status_probability'];exact=observe(qc,state)
    details={'config':asdict(cfg),'steps':4,'profile':'baseline','reference':ref['rows'],
        'exact_evolution':info,'exact_observables':exact,'max_joint_error':joint_error,'MPS':numeric,
        'MPS_vs_reference_absolute_errors':[abs(a-b) for a,b in zip(p,ref['rows'][-1]['probabilities'])],
        'work_bit_probabilities':[numeric['data'][f'work_{i}'][1] for i in range(17)],
        'checkpoint_status_max_error':max(abs(a-b) for seen,want in zip(info['checkpoints'],ref['rows']) for a,b in zip(seen['observables']['probabilities'],want['probabilities'])),
        'checkpoint_work_max':max(c['observables']['work_nonzero'] for c in info['checkpoints'])}
    passed=not failures and joint_error<2e-11 and details['checkpoint_work_max']<2e-11 and max(details['MPS_vs_reference_absolute_errors'])<2e-11 and math.fsum(details['work_bit_probabilities'])<2e-11 and all(r['history_max_absolute_error']<2e-11 and r['work_union_upper_bound']<2e-11 for r in comparisons)
    save('tau19_work_diagnostic.json',{'status':'passed' if passed else 'failed','hypothesis':'Index/extraction mismatch or residual predicate data at full physical-step boundaries',
        'mapping_scope':'Main work[4]=original143 -> Aer active132; material slab flag and boundary y==0 flag. ROI work9[2] is unused.',
        'truth_table':{'valid_positions':17*9*9,'failures':failures,'method':'actual X/controlled-X integer permutation; material/boundary/ROI copied to fresh history then uncomputed'},
        'actual_gate_MPS_crosschecks':comparisons,'short_transport':details,
        'main_T15_resolved':False,'limitations':'T4 covers entry and subsequent scattering branches at the slab front, not late T15 entanglement/truncation. Basis predicate checks do not bound main response error.',
        'budgets':{'max_exact_states':60000,'MPS_bond':128,'wall_deadline':None},'sources':source_manifest()})
    print(json.dumps({'work_diagnostic':passed,'joint_error':joint_error,'T4_work':math.fsum(details['work_bit_probabilities'])}),flush=True)

class GateBudgetExceeded(RuntimeError):
    status='budget_limited'

def transport_iqae():
    import numpy as np
    from iqae import OracleSpec, build_grover_operator_for_spec, estimate_relative_iqae
    base=load_model('q_mps_one_cell'); main=load_model('compact_n8_w0002_tau19')
    candidates=[('uniform_two_cell',load_model('q_uniform_small'),3),
                ('tau19_one_cell',replace(base,mats=main.mats,h=.5),2)]
    screen=[]
    for name,cfg,steps in candidates:
        result=reference(cfg,steps);screen.append({'id':name,'config':asdict(cfg),'steps':steps,'probabilities':result['rows'][-1]['probabilities'],
          'selected':True,'reason':'Natural short-step transport; no external rarity factor. One retains the tau19 material interaction probabilities.'})
    save('transport_IQAE_selection.json',{'candidates':screen,'selection_before_random_data':True,'fixed_seeds':[2026110701,2026110702],
        'scope':'These reduced geometries are validation subcases, not replacements for the fixed main candidates.'})
    for index,(name,cfg,steps) in enumerate(candidates):
        begin=time.perf_counter(); cpu=time.process_time(); A,_=build_unitary_circuit_qiskit(cfg,steps,kernel_profile='baseline')
        regs=registers(A); Q,obj,_=build_grover_operator_for_spec(A_circuit=A,status_qubits=regs['status'],g_qubits=regs['g'],spec=OracleSpec(good='detect'))
        marked=Q.state_preparation; bit=obj[0]
        Aops=list(flattened(marked)); Qops=list(flattened(Q))
        budgets={'max_states':60000,'max_actual_primitive_gate_applications':2000000,'shots':20000,'queries':10000000,'max_power':128,'wall_clock_deadline':None}
        used=0; cache={}; evolution=[]
        def simulate(ops,state):
            nonlocal used
            if used+len(ops)>budgets['max_actual_primitive_gate_applications']:raise GateBudgetExceeded('actual gate application budget exhausted')
            peak=len(state)
            for op in ops:
                state=apply_instruction(state,op);used+=1;peak=max(peak,len(state))
                if len(state)>budgets['max_states']:raise GateBudgetExceeded('independent complex-state budget exhausted')
            return state,peak
        def probability(k):
            if k in cache:return cache[k]
            start=time.perf_counter(); before=used
            if used+len(Aops)+k*len(Qops)>budgets['max_actual_primitive_gate_applications']:raise GateBudgetExceeded('actual gate application budget exhausted before power construction/evolution')
            state,peak=simulate(Aops,{0:1+0j})
            for _ in range(k):state,peak0=simulate(Qops,state);peak=max(peak,peak0)
            norm=math.fsum(abs(a)**2 for a in state.values());p=math.fsum(abs(a)**2 for key,a in state.items() if (key>>bit)&1)
            if not math.isfinite(p) or not 0<=p<=1 or abs(norm-1)>2e-11:raise RuntimeError(f'invalid actual gate probability or norm: p={p}, norm={norm}')
            cache[k]=p;evolution.append({'power':k,'probability':p,'raw_norm':norm,'norm_residual':abs(norm-1),'seconds':time.perf_counter()-start,
                'primitive_gate_applications':used-before,'peak_basis':peak,'final_basis':len(state)})
            return p
        for k in (0,1,2):probability(k)
        # A and A^-1 are actual transport operators; do not use classical p to construct either.
        state,_=simulate(list(flattened(A)),{0:1+0j})
        state,_=simulate(list(flattened(A.inverse())),state)
        inverse_error=math.fsum(abs(a-(1 if key==0 else 0))**2 for key,a in state.items())
        rng=np.random.default_rng(2026110701+index);counts=[]
        def sample_counts(k,n):
            q=probability(k);hits=int(rng.binomial(n,q));counts.append({'power':k,'shots':n,'success_count':hits,'actual_gate_probability':q});return hits
        result=estimate_relative_iqae(sample_counts,relative_error=.01,confidence=.95,shots=512,max_shots=20000,max_queries=10000000,max_power=128)
        # Reference and analytic amplification are used only after the complete run.
        p_ref=screen[index]['probabilities'][1];theta=math.asin(math.sqrt(p_ref))
        for item in evolution:
            formula=math.sin((2*item['power']+1)*theta)**2
            item.update(ideal_formula_for_comparison=formula,amplified_probability_absolute_difference=abs(item['probability']-formula))
        estimate=result['estimate'];interval=result['confidence_interval']
        save(f'transport_IQAE_{name}.json',{'evidence_kind':'real transport actual-gate complex-amplitude evolution and Bernoulli measurement sampling',
            'config':asdict(cfg),'steps':steps,'profile':'baseline','objective':'detect','seed':2026110701+index,
            'fingerprints':{'A':circuit_fingerprint(A),'A_inverse':circuit_fingerprint(A.inverse()),'marked_A':circuit_fingerprint(marked),'Q':circuit_fingerprint(Q)},
            'qubits':marked.num_qubits,'primitive_components':{'marked_A':len(Aops),'Q':len(Qops)},'budgets':budgets,
            'actual_primitive_gate_applications':used,'A_inverse_state_error_squared':inverse_error,'actual_evolutions':evolution,
            'measurement_batches':counts,'estimator':result,'independent_classical_reference':p_ref,
            'posthoc':{'relative_error':None if estimate is None else abs(estimate-p_ref)/p_ref,
                'interval_contains_reference':False if interval is None else interval[0]<=p_ref<=interval[1]},
            'reference_error':'independent binary64 recurrence; floating arithmetic, not an interval enclosure',
            'simulation_error':'no state pruning; only exactly zero amplitudes removed. Formula/inverse/norm checks quantify observed error, not a certified roundoff bound.',
            'full_pipeline_certified':False,'empirical_coverage_validated':False,'main_rare_IQAE_completed':False,
            'seconds':time.perf_counter()-begin,'process_cpu_seconds':time.process_time()-cpu,'versions':software_versions(),'sources':source_manifest()})
        print(json.dumps({'IQAE':name,'status':result['status'],'p_ref':p_ref,'estimate':estimate,'actual_gates':used,'costs':result['costs']}),flush=True)

def control():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from validation import _probe_backend_options
    import re
    folder=ROOT/'outputs/direct_response/baseline_tau19_T15_bond128_01'
    raw=read(folder/'n8_w0002_tau19_T15_bond128_trunc1e-16.json')
    cfg=load_model('compact_n8_w0002_tau19')
    A,_=build_unitary_circuit_qiskit(cfg,15,kernel_profile='baseline')
    compiled=transpile(A,**raw['identity']['compile']);obs=compiled.copy();regs=registers(compiled)
    obs.save_probabilities(regs['status'],label='status_probability')
    for i,bit in enumerate(regs['work']):obs.save_probabilities([bit],label=f'work_{i}')
    fingerprints={'original_A_fingerprint':circuit_fingerprint(A),'compiled_A_fingerprint':circuit_fingerprint(compiled),'diagnostic_fingerprint':circuit_fingerprint(obs)}
    options=dict(raw['backend_options']);options['matrix_product_state_max_bond_dimension']=256
    # Validate options against the installed backend; do not run it.
    backend=AerSimulator(**options)
    changes=[k for k in options if options[k]!=raw['backend_options'].get(k)]
    log=raw['metadata']['MPS_log_data'];end=log.rfind('BD=[');last=log[end+4:log.index(']',end)]
    last_dims=list(map(int,last.split()))
    mapping=dict(raw['metadata']['input_qubit_map'])
    save('tau19_b256_control_preparation.json',{'execution_status':'prepared_not_executed','hypothesis':'Bond cap / accumulated MPS approximation contributes to the isolated work[4] leakage; no mapping or small-case predicate defect reproduced.',
        'original_evidence':str(folder.relative_to(ROOT)).replace('\\','/'),'control_bond':256,
        'only_backend_change':changes,'backend_options':options,'actual_supported_bond':backend.options.matrix_product_state_max_bond_dimension,
        'circuit_identity_rebuilt_matches':{k:v==raw[k] for k,v in fingerprints.items()},'fingerprints':fingerprints,
        'register_map_matches':regs==raw['register_map'],'work4_original_bit':regs['work'][4],'work4_active_bit_in_return':mapping[regs['work'][4]],
        'observation_instructions':[(i.operation.label,[obs.find_bit(q).index for q in i.qubits]) for i in obs.data if i.operation.name.startswith('save_')],
        'compile':raw['identity']['compile'],'final_logged_bond_dimensions':last_dims,
        'final_bonds_at_cap':sum(d==128 for d in last_dims),'max_final_logged_bond':max(last_dims),
        'comparison_protocol':{'held_fixed':['model','T15','baseline','compile u/cx opt0 seed917','seed20261030','double','trunc1e-16','zero/chop0','Aer1/MPS1/BLAS1','LAPACKfalse','countsfalse','same observations and MPS logging'],
          'record_separately':['raw S/D/A/B','D relative error <=1%','all work probabilities and union upper bound','raw probability sum','strict 2e-11 diagnostics','unresolved issues','CPU/wall/peak memory'],
          'interpretation':'A clear leakage reduction with stable D supports bond sensitivity at these settings; unchanged leakage weakens that hypothesis and leaves backend/accumulated roundoff unresolved. One contrast is not an asymptotic error bound and triggers no further automatic jobs.',
          'quantitative_diagnostic':'Report work4_control/work4_original and residual_control/residual_original; ratios <=0.1 are predeclared strong sensitivity evidence, >0.1 inconclusive. Separate strict work <=2e-11 and D <=1% decisions.'},
        'resource_budget':{'CPU':1,'Aer_threads':1,'MPS_threads':1,'BLAS_threads':1,'memory_MiB':8192,'wall_deadline':None},
        'no_main_state_evolution':True})
    print(json.dumps({'control_ready':changes==['matrix_product_state_max_bond_dimension'] and all(fingerprints[k]==raw[k] for k in fingerprints),'last_bonds_at_cap':sum(d==128 for d in last_dims)}),flush=True)

def classical_scope():
    old=read('outputs/revision_screening/20261030_compact_deep_01/compact_screen.json')
    cfg=load_model('absorption_AB');start=time.perf_counter();r=reference(cfg,9)
    p=r['rows'][-1]['probabilities'][1];expected=math.exp(-3)
    save('classical_scope.json',{'purpose':'Separate reused tail/grid evidence and verify a comparable pure-absorption external-code starting case',
        'reused_tail_rows':[{k:r[k] for k in ('id','T','p_D','p_S','tail_relative_bound','reachable_states','transition_nnz','elapsed_seconds','raw_result')} for r in old['rows'] if r['id'] in [f'compact_n8_w0002_tau{t}' for t in (14,19,23)]],
        'reused_grid_rows':[r for r in old['refinements'] if r['base_cells']==8 and r['omega']==.002],
        'external_comparison_start':{'config':asdict(cfg),'steps':9,'probabilities':r['rows'][-1]['probabilities'],
            'reference_D':p,'analytic_D':expected,'absolute_difference':abs(p-expected),'seconds':time.perf_counter()-start,
            'analytic_parameters':{'Sigma_A':1,'Sigma_B':2,'thickness_A':1,'thickness_B':1,'omega':0,'total_optical_thickness':3},
            'external_code_executed':False,'matched_scope':'Pure absorption with normal collimated beam; external detector contains the entire unscattered beam. No claim about six-direction scattering vs continuous transport.',
            'prospective_fixed_history_budget':1000000,'prospective_seed':20261107,
            'prospective_interval_criterion':'single fixed-sample 95% exact binomial interval [L,U]; L>0 and (U-L)/(2L)<=0.01; independent analytic p must be in interval for this crosscheck. No adaptive additional histories.'},
        'new_fine_grid_calculations':False,'new_quantum_evolution':False})
    print(json.dumps({'pure_absorption_reference':p,'analytic_relative_difference':abs(p-expected)/expected}),flush=True)

def finalize():
    import csv
    from datetime import datetime, timezone
    audit=read(OUT/'direct_response_audit.json')
    assert audit['all_integrity_checks_passed']
    assert all(sha(ROOT/k)==v['sha256'] and (ROOT/k).stat().st_size==v['bytes'] for k,v in audit['original_file_manifest'].items())
    assert source_manifest()==audit['current_sources'], 'Production PY changed during closeout'
    work=read(OUT/'tau19_work_diagnostic.json');control_result=read(OUT/'tau19_b256_control_preparation.json')
    iqae_rows=[read(OUT/f'transport_IQAE_{name}.json') for name in ('uniform_two_cell','tau19_one_cell')]
    tests=read(OUT/'focused_regressions.json');verification=read('tests/last_verification.json')
    copy=verification['root_copy_acceptance']
    assert copy['passed'] and tests['returncode']==0
    assert all(sha(ROOT/k)==v for k,v in copy['tested_sources'].items())
    table=ROOT/'outputs/direct_response/results.csv'
    with table.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;table_rows=list(reader)
    added=['aer_threads','mps_threads','blas_threads','mps_lapack','precision','mps_trunc','zero_threshold','chop_threshold','source_identity_hash']
    fields=fields+[k for k in added if k not in fields]
    for record in audit['rows']:
        tau=record['tau'];target=next(r for r in table_rows if r['profile']=='baseline' and int(r['tau'])==tau)
        raw=read(record['source']);sup=record['supervision'];backend=record['backend_options']
        target.update(dataset_id=f'baseline_tau{tau}_T15_bond{record["bond"]}_01',evidence_kind='raw_complete',execution_status='complete')
        for i,label in enumerate('SDAB'):
            target['p_'+label]=record['probabilities'][i];target['ref_'+label]=record['reference'][i];target['abs_error_'+label]=record['absolute_errors'][i]
        for k in ('raw_probability_sum','normalization_residual','D_relative_error','D_relative_error_percent'):target[k]=record[k]
        target['work_upper_bound']=record['work_union_upper_bound']
        for k in ('rare_accuracy_status','strict_status','numerical_validity_status','scientific_certification_passed'):target[k]=raw[k]
        target.update(tree_cpu_seconds=sup['tree_cpu_seconds'],wall_seconds=sup['elapsed_seconds'],average_cores=sup['cpu_average_cores'],
          **record['fingerprints'],source_path=record['source'].removeprefix('outputs/'),source_pointer='/',
          aer_threads=backend['max_parallel_threads'],mps_threads=backend['mps_omp_threads'],blas_threads=raw['blas_threads'],
          mps_lapack=backend['mps_lapack'],precision=backend['precision'],mps_trunc=raw['truncation_threshold'],
          zero_threshold=backend['zero_threshold'],chop_threshold=backend['chop_threshold'],source_identity_hash=raw['identity']['identity_hash'])
    assert len(table_rows)==7 and len({r['dataset_id'] for r in table_rows})==7
    with table.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(table_rows)
    catalog=read('outputs/data_catalog.json');outbase=ROOT/'outputs'
    def dataset(identifier,category,title,primary,kind,scope,limits,files):
        item={'dataset_id':identifier,'category':category,'title':title,'primary_path':primary,'files':sorted(files),
              'evidence_kind':kind,'execution_status':'complete','scientific_scope':scope,'limitations':limits,'manuscript_figure_or_table':None}
        old=next((x for x in catalog['datasets'] if x['dataset_id']==identifier),None)
        if old is None:catalog['datasets'].append(item)
        else:old.update(item)
    for tau,bond in ((14,64),(19,128)):
        identifier=f'baseline_tau{tau}_T15_bond{bond}_01';folder=outbase/'direct_response'/identifier
        files=[p.relative_to(outbase).as_posix() for p in folder.rglob('*') if p.is_file()]
        dataset(identifier,'direct_response',f'baseline τ{tau}/T15/bond{bond} 云端直接响应',folder.relative_to(outbase).as_posix(),
                'full_transport_MPS_direct_probability','固定离散模型D响应；原始四态、逐work、三重电路身份、后端及监督记录',
                'D通过1%；严格四态失败；'+('数值检查passed但不代表全分布通过' if tau==14 else '数值有效性unresolved')+'；不代表IQAE或物理连续极限',files)
        original=read(folder/'resolved_request.json').get('outdir')
        archive={'id':identifier,'source':'user supplied directly in outputs; no external source directory assumed',
                 'archive':folder.relative_to(outbase).as_posix(),'original_recorded_outdir':original,
                 'files':{p.relative_to(folder).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.rglob('*') if p.is_file()},
                 'original_import_copy_verified':False,'read_only_audit_unchanged':True}
        old=next((a for a in catalog['archived_returns'] if a['id']==identifier),None)
        if old is None:catalog['archived_returns'].append(archive)
        else:old.update(archive)
    prefix=OUT.relative_to(outbase).as_posix()
    dataset('baseline_freeze_20261107','reference_and_validation','三档归档复核与τ19工作位有限定位',prefix,
            'read_only_evidence_audit_and_actual_gate_diagnostics',
            '48份原件一致性、独立经典复核、1377点谓词、13点真实门/MPS、τ19原几何T4；控制电路仅构建',
            'T15异常未解决；bond256对照未演化；工作位/归一残差不是D或IQAE误差界',
            [prefix+'/'+f for f in ('direct_response_audit.json','tau19_work_diagnostic.json','tau19_b256_control_preparation.json','focused_regressions.json','classical_scope.json')])
    dataset('transport_IQAE_short_cases_20261107','iqae','两个自然输运子例的实际门IQAE',prefix,
            'actual_transport_gate_evolution_and_measurement_counts',
            '独立复振幅引擎实际演化A、A逆、Q及估计器自适应所用power，固定seed及预算，单次过程1%/95%停止',
            '非主例；一次运行不验证覆盖率；浮点模拟未作区间误差认证；不声明端到端总1%',
            [prefix+'/transport_IQAE_'+name+'.json' for name in ('selection','uniform_two_cell','tau19_one_cell')])
    catalog['pending_main_data']=[]
    catalog['unresolved_scientific_items']=['tau19 work[4] and normalization mechanism','tau23 normalization mechanism',
        'certified amplified-circuit numerical error','main rare transport full IQAE','matched mature-code physics comparison','continuum discretization validation','fair end-to-end speedup evidence']
    catalog['updated_utc']=datetime.now(timezone.utc).isoformat()
    catalog['data_dictionary'].update({'aer_threads/mps_threads/blas_threads':'Actual recorded requests; not measured occupied cores. tau23 used MPS50; tau14/19 used MPS1.',
      'mps_lapack/precision/mps_trunc/zero_threshold/chop_threshold':'Per-run numerical settings; independent of IQAE relative precision/confidence.',
      'source_identity_hash':'Original immutable v2 run identity, not recomputed with current sources.'})
    paths={p for d in catalog['datasets'] for p in d['files']}|{catalog['direct_response_table']}
    catalog['file_manifest']={p:{'bytes':(outbase/p).stat().st_size,'sha256':sha(outbase/p)} for p in sorted(paths)}
    (outbase/'data_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    columns=['category','dataset_id','title','primary_path','evidence_kind','execution_status','scientific_scope','limitations']
    with (outbase/'data_catalog.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(catalog['datasets'])
    summary={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'production_root_PY_unchanged':True,
        'production_sources':source_manifest(),'CMD_change':'Only SUITE, BOND and OUTDIR default values; same grouped launcher and setup_qc.cmd',
        'read_only_raw_files_verified':len(audit['original_file_manifest']),'audit_integrity_checks':sum(len(r['checks']) for r in audit['rows']),
        'audit_all_passed':audit['all_integrity_checks_passed'],'direct_D_all_passed':all(r['D_relative_error']<=.01 for r in audit['rows']),
        'original_numerical_status':{str(r['tau']):r['original_labels'] for r in audit['rows']},
        'work_diagnostic_passed':work['status']=='passed','main_work_anomaly_resolved':False,
        'actual_transport_IQAE':[{'id':name,'status':r['estimator']['status'],'costs':r['estimator']['costs'],'posthoc':r['posthoc'],'full_pipeline_certified':False} for name,r in zip(('uniform_two_cell','tau19_one_cell'),iqae_rows)],
        'focused_regressions':{'returncode':tests['returncode'],'tests':27,'evidence':prefix+'/focused_regressions.json'},
        'classical_scope':prefix+'/classical_scope.json',
        'root_copy_acceptance':{'passed':copy['passed'],'checks':len(copy['checks']),'tested_sources_match_current':True},
        'cloud_control':{'prepared_only':True,'dry_run':copy['details']['cloud_dry_request'],'identity_checks':control_result['circuit_identity_rebuilt_matches']},
        'frozen_scope':['production physical kernel and compilation contract','fixed-model T15 direct-D comparison at each recorded setting','existing statistical protocol','small actual gate connection and resource-count definitions'],
        'not_frozen':['full-state MPS numerical validity','main rare full IQAE','end-to-end certified total 1%','continuous physical model accuracy','quantum advantage'],
        'no_cloud_execution':True,'no_main_MPS_re_evolution':True,'dependency_updates':False,
        'new_evidence_directory':str(OUT.relative_to(ROOT)).replace('\\','/'),'catalog':{'datasets':len(catalog['datasets']),'files':len(paths),'categories':3},
        'collector_sha256':sha(Path(__file__)),'readme_sha256':sha(ROOT/'docs/README.md')}
    verification['baseline_freeze_20261107']=summary
    verification['current_evidence_section']='baseline_freeze_20261107; earlier sections are historical snapshots'
    verification['tested_input_scope']='Previous grouped regression snapshot retained; current affected checks and root-copy source identity are in baseline_freeze_20261107.'
    verification['tested_inputs_match_current_source']=False
    if 'next_manual_tasks' in verification.get('precision_revision',{}):
        verification['precision_revision']['next_manual_tasks_status']='Historical preparation superseded by baseline_freeze_20261107; three baseline direct responses are returned.'
    verification={'current_evidence_section':verification['current_evidence_section'],
                  'baseline_freeze_20261107':summary,**verification}
    (ROOT/'tests/last_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'catalog_datasets':len(catalog['datasets']),'catalog_files':len(paths),'direct_rows':len(table_rows),'raw_unchanged':True,'verification_updated':True}),flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    try:
        {'audit':audit,'work':work_diagnostic,'iqae':transport_iqae,'control':control,'classical':classical_scope,'finalize':finalize}[stage]()
    except Exception as exc:
        import traceback
        save(stage+'_failure.json',{'status':getattr(exc,'status','failed'),'error':repr(exc),'traceback':traceback.format_exc(),'sources':source_manifest()})
        raise
