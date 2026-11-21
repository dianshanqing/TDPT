"""Rare probability criteria and bounded, actual compiled-gate diagnostics."""
import json
import math
from pathlib import Path
import time
import traceback

LEGACY_ATOL=2e-11


def iqae_component_cost(components, rounds):
    """Compositional cost of an observed schedule; never expands deep Q powers.

    Atilde includes the objective marker. Component gates concatenate without
    cross-component optimization. Depth sum is an upper bound; no runtime or
    MPS feasibility is inferred from these static counts.
    """
    from ae_likelihood import integer
    shots = grover = peak = 0
    for row in rounds:
        n, k = integer(row['shots'], 'shots'), integer(row['power'], 'power')
        if n <= 0 or k < 0:
            raise ValueError('positive shots and nonnegative power required')
        shots += n; grover += n*k; peak = max(peak,k)
    calls={'Atilde':shots+grover,'Atilde_inverse':grover,'S0':grover,'Sf':grover}
    return {'shots':shots,'max_power':peak,'A_calls':shots+grover,'A_inverse_calls':grover,
        'queries_A_and_inverse':shots+2*grover,'zero_reflection_calls':grover,
        'objective_reflection_calls':grover,'measurement_calls':shots,
        'gates_excluding_measurement':sum(calls[key]*components[key]['gates'] for key in calls),
        'cx':sum(calls[key]*components[key]['cx'] for key in calls),
        'max_single_circuit_depth_upper_bound':components['Atilde']['depth']+
            peak*sum(components[key]['depth'] for key in calls)+1,
        'qubits':max(components[key]['qubits'] for key in calls),
        'cost_scope':'concatenated compiled components; depth upper bound includes final measurement; no whole-circuit optimization or runtime estimate'}


def validate_relative_iqae_statistics(outdir, reference_file, *, repetitions=1000, seed=20261106):
    """Prespecified ideal measurement experiment; not transport/MPS evidence."""
    import hashlib
    import numpy as np
    from scipy.stats import beta
    from iqae import estimate_relative_iqae
    from runtime_identity import write_json_new,source_manifest,software_versions
    from result_storage import content_digest, read_json
    output=Path(outdir); output.mkdir(parents=True,exist_ok=True)
    source=Path(reference_file); saved=read_json(source)
    cases=[]
    for tau in (14,19,23):
        name=f'compact_n8_w0002_tau{tau}'
        matches=[r for r in saved['rows'] if r['id']==name and r['steps']==15]
        if len(matches)!=1:raise ValueError('reference identity mismatch: '+name)
        cases.append((name,float(matches[0]['p_D_float64'])))
    cases += [('moderate_0p2',.2),('moderate_0p7',.7)]
    protocol={'label':'理想测量模型下的估计器统计验证','repetitions_per_case':repetitions,'seed':seed,
        'relative_error':.01,'confidence':.95,'shots':512,'max_shots':200000,
        'max_queries':100000000000,'max_power':10000000,'cases':cases,
        'reference_path':str(source),'reference_sha256':content_digest(source)['sha256'],
        'random_stream':'SeedSequence(seed).spawn(5*repetitions), fixed before observations',
        'coverage_claim':'finite repetitions diagnose; theoretical guarantee is alpha-spending CP, single process',
        'versions':software_versions(),'sources':source_manifest()}
    write_json_new(output/'protocol.json',protocol)
    streams=np.random.SeedSequence(seed).spawn(len(cases)*repetitions); summaries=[]
    def cp(k,n):
        return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
                1. if k==n else float(beta.isf(.025,k+1,n-k))]
    for case_index,(name,p) in enumerate(cases):
        rows=[]; started=time.perf_counter()
        # The generator closes over truth. The estimator only receives counts.
        for repetition in range(repetitions):
            rng=np.random.default_rng(streams[case_index*repetitions+repetition])
            def sample(k,n):
                q=math.sin((2*k+1)*math.asin(math.sqrt(p)))**2
                return int(rng.binomial(n,q))
            result=estimate_relative_iqae(sample)
            result.update(repetition=repetition,spawn_key=list(streams[case_index*repetitions+repetition].spawn_key))
            rows.append(result)
        complete=[r for r in rows if r['status']=='completed']
        covered=sum(r['confidence_interval'][0]<=p<=r['confidence_interval'][1] for r in complete)
        accurate=sum(abs(r['estimate']-p)/p<=.01 for r in complete)
        errors=np.array([r['estimate']-p for r in complete])
        summary={'case':name,'reference':p,'label':protocol['label'],'repetitions':repetitions,
            'completed':len(complete),'coverage_count':covered,'coverage_fraction':covered/repetitions,
            'coverage_binomial_95pct_interval':cp(covered,repetitions),
            'within_1pct_count':accurate,'within_1pct_fraction':accurate/repetitions,
            'within_1pct_binomial_95pct_interval':cp(accurate,repetitions),
            'bias_completed':float(errors.mean()) if len(errors) else None,
            'relative_bias_completed':float(errors.mean()/p) if len(errors) else None,
            'relative_rmse_completed':float(np.sqrt(np.mean((errors/p)**2))) if len(errors) else None,
            'budget_exhausted_fraction':sum(r['status']=='budget_limited' for r in rows)/repetitions,
            'invalid_or_failed_fraction':sum(r['status'] not in ('completed','budget_limited') for r in rows)/repetitions,
            'seconds':time.perf_counter()-started,'costs':{}}
        for key in ('shots','queries_A_and_inverse','max_power','A_calls','A_inverse_calls','grover_iterations'):
            vals=np.array([r['costs'][key] for r in rows])
            summary['costs'][key]={'min':int(vals.min()),'median':float(np.median(vals)),
                'p95':float(np.quantile(vals,.95)),'max':int(vals.max()),'mean':float(vals.mean())}
        write_json_new(output/(name+'.json'),{'summary':summary,'runs':rows})
        summaries.append(summary); print(json.dumps(summary),flush=True)
    report={'protocol':protocol,'summaries':summaries,'status':'completed',
            'full_transport_or_MPS_execution':False,'simultaneous_five_case_95pct_claim':False}
    write_json_new(output/'summary.json',report)
    return report


def validate_relative_iqae_transport(outdir, *, seed=20261106):
    """Actual sparse gate evolution of transport A and Q, then sampled counts."""
    import numpy as np
    from iqae import OracleSpec,build_grover_operator_for_spec,estimate_relative_iqae
    from models import load_model
    from reference_solver import reference
    from unitary_circuit import build_unitary_circuit_qiskit
    from reference_engine import evolve
    from runtime_identity import write_json_new,circuit_fingerprint,source_manifest,software_versions
    output=Path(outdir);output.mkdir(parents=True,exist_ok=True)
    config='q_mps_one_cell';steps=2;cfg=load_model(config)
    a,_=build_unitary_circuit_qiskit(cfg,steps,kernel_profile='baseline')
    regs={r.name:r for r in a.qregs}
    q,obj,_=build_grover_operator_for_spec(A_circuit=a,status_qubits=regs['status'],g_qubits=regs['g'],spec=OracleSpec())
    state,runtime=evolve(q.state_preparation,max_states=60000)
    # Independently execute actual Q gates, including A inverse, zero reflection,
    # A and objective reflection. No sin² law is used to generate these counts.
    current_power=0;measurements=[];rng=np.random.default_rng(seed)
    def sample(k,n):
        nonlocal state,current_power
        while current_power<k:
            state,detail=evolve(q,state,max_states=60000);current_power+=1
        probability=math.fsum(abs(v)**2 for index,v in state.items() if (index>>obj[0])&1)
        norm=math.fsum(abs(v)**2 for v in state.values())
        if not 0<=probability<=1 or abs(norm-1)>2e-11:
            raise ValueError('raw sparse gate evolution violates probability/norm check')
        hits=int(rng.binomial(n,probability))
        measurements.append({'power':k,'shots':n,'success_count':hits,'raw_probability':probability,
                             'raw_norm':norm,'basis_states':len(state)})
        return hits
    result=estimate_relative_iqae(sample,max_power=128,max_queries=10000000,max_shots=20000)
    p=reference(cfg,steps)['rows'][-1]['probabilities'][1]
    for row in measurements:
        exact=math.sin((2*row['power']+1)*math.asin(math.sqrt(p)))**2
        row['grover_relation_absolute_error']=abs(row['raw_probability']-exact)
    result.update(label='真实输运小电路与相对 IQAE 连接验证',config=config,steps=steps,
        profile='baseline',seed=seed,reference_posthoc_only=p,measurements=measurements,
        posthoc_relative_error=None if result['estimate'] is None else abs(result['estimate']-p)/p,
        A_fingerprint=circuit_fingerprint(a),A_inverse_fingerprint=circuit_fingerprint(a.inverse()),
        marked_A_fingerprint=circuit_fingerprint(q.state_preparation),Q_fingerprint=circuit_fingerprint(q),
        qubits=q.num_qubits,all_qubits_reflected=list(q.reflection_qubits)==list(range(q.num_qubits)),
        simulation='Unpruned sparse double gate evolution; Born binomial sampling of actual Q^k A states',
        sampling_note='Cached evolved state per power is equivalent to fresh noiseless circuit preparations; counts use independent RNG draws',
        exact_path_scope='No bond or amplitude truncation; floating arithmetic is checked, not rigorously interval bounded',
        observed_coverage_claim=False,initial_evolution=runtime,versions=software_versions(),sources=source_manifest())
    write_json_new(output/'transport_iqae.json',result)
    print(json.dumps({k:result[k] for k in ('status','estimate','confidence_interval','costs','posthoc_relative_error')}),flush=True)
    return result


def event_pruning_bound(observed_probability, discarded_l2_sum):
    return 2*math.sqrt(max(0.,observed_probability))*discarded_l2_sum+discarded_l2_sum**2


def evolve_with_pruning_bound(circuit, amplitude_cutoff=1e-15, max_states=60000):
    """Diagnostic approximation only; ledger the norm removed after every gate.

    Unitary propagation preserves earlier L2 error, giving delta <= sum ||drop||.
    Floating gate/matrix arithmetic is NOT covered by this pruning-only bound.
    The retained state is never renormalized and production gates are unchanged.
    """
    from reference_engine import flattened,apply_instruction
    from qiskit.circuit.library import UnitaryGate
    if not math.isfinite(amplitude_cutoff) or amplitude_cutoff < 0:
        raise ValueError('amplitude_cutoff must be finite and nonnegative')
    if any(ins.operation.name=='u' for ins in circuit.data):
        original=circuit;circuit=original.copy_empty_like();circuit.global_phase=original.global_phase
        for ins in original.data:
            op=UnitaryGate(ins.operation.to_matrix()) if ins.operation.name=='u' else ins.operation
            circuit.append(op,[circuit.qubits[original.find_bit(q).index] for q in ins.qubits])
    state={0:1+0j};discarded=0.;events=0;removed_count=0;peak=1;gates=0
    start=time.perf_counter()
    for item in flattened(circuit):
        state=apply_instruction(state,item);gates+=1;peak=max(peak,len(state))
        dropped=[k for k,a in state.items() if abs(a)<amplitude_cutoff]
        if dropped:
            discarded+=math.sqrt(math.fsum(abs(state[k])**2 for k in dropped));events+=1;removed_count+=len(dropped)
            for k in dropped:del state[k]
        if len(state)>max_states:raise MemoryError(f'pruned sparse state budget exceeded: {len(state)} > {max_states}')
    return state,{'seconds':time.perf_counter()-start,'amplitude_cutoff':amplitude_cutoff,
        'discarded_l2_sum':discarded,'pruning_events':events,'discarded_amplitudes':removed_count,
        'peak_basis_before_pruning':peak,'retained_basis':len(state),'primitive_gates':gates,'max_states':max_states,
        'bound_scope':'Pruning only assuming unitary mathematical gate propagation; excludes floating arithmetic error. No renormalization.', 'deadline':None}


def run_bounded_sparse_audit(spec,outdir,cpu_budget=1):
    from models import load_model,resolve_path
    from unitary_circuit import build_unitary_circuit_qiskit
    from reference_solver import reference
    from reference_engine import observe
    from qiskit import transpile
    from runtime_identity import write_json_new,circuit_fingerprint,source_manifest,software_versions
    output=resolve_path(outdir);output.mkdir(parents=True,exist_ok=True);rows=[]
    for case in spec['cases']:
        cfg=load_model(case['config']);steps=case['steps']
        ref=reference(cfg,steps)['rows'][-1]['probabilities']
        original,_=build_unitary_circuit_qiskit(cfg,steps,kernel_profile='compressed')
        compiled=transpile(original,basis_gates=['u','cx'],optimization_level=0,seed_transpiler=917,num_processes=1)
        for stage,circuit in [('original',original),('compiled',compiled)]:
            print(json.dumps({'stage':'bounded_sparse_start','case':case['id'],'circuit':stage}),flush=True)
            try:
                state,runtime=evolve_with_pruning_bound(circuit,spec.get('amplitude_cutoff',1e-15),spec.get('max_states',60000))
                actual=observe(circuit,state);comp=compare_rare_probability(actual['probabilities'][1],ref[1])
                bound=event_pruning_bound(actual['probabilities'][1],runtime['discarded_l2_sum'])
                ratio=bound/ref[1] if ref[1] else None
                row={'status':'complete','case':case,'stage':stage,'fingerprint':circuit_fingerprint(circuit),
                    'runtime':runtime,'observed':actual,'reference':ref,'comparison':comp,
                    'pruning_pD_absolute_bound':bound,'pruning_pD_relative_bound':ratio,
                    'passes_1pct_including_pruning':(comp['relative_error']+ratio<=.01 if ref[1] else None)}
            except Exception as exc:row={'status':'failed','case':case,'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()}
            write_json_new(output/(case['id']+'_'+stage+'.json'),row);rows.append(row)
            print(json.dumps({'case':case['id'],'stage':stage,'status':row['status'],'relative_error':row.get('comparison',{}).get('relative_error'),'pruning_relative_bound':row.get('pruning_pD_relative_bound')}),flush=True)
    report={'status':'complete','rows':rows,'sources':source_manifest(),'versions':software_versions(),
        'passed':all(r.get('passes_1pct_including_pruning') for r in rows),'deadline':None,
        'scope':'Bounded-pruning floating approximation, not unpruned exact or interval-certified gate evolution.'}
    write_json_new(output/'bounded_sparse_audit.json',report)
    return report


def compare_rare_probability(observed, reference):
    """Fixed-model response accuracy; missing output is never a zero event.

    The legacy 5% and absolute fields are retained only for historical readers.
    They cannot satisfy the formal 1% criterion.
    """
    reference=float(reference)
    if not math.isfinite(reference) or not 0 <= reference <= 1:
        raise ValueError('reference must be finite and in [0,1]')
    if observed is None:
        return {'reference':reference,'observed':None,'result_status':'missing',
            'absolute_error':None,'relative_error':None,'legacy_absolute_tolerance':LEGACY_ATOL,
            'passes_legacy_absolute':False,'passes_formal_1pct':False,
            'passes_pilot_5pct':False,'passes_zero_reference':False if reference==0 else None,
            'formal_absolute_error_budget':.01*reference if reference else None}
    observed=float(observed)
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (observed,reference)):
        raise ValueError('probabilities must be finite and in [0,1]')
    error=abs(observed-reference)
    relative=error/reference if reference else None
    return {'reference':reference,'observed':observed,'result_status':'available','absolute_error':error,
        'relative_error':relative,'legacy_absolute_tolerance':LEGACY_ATOL,
        'passes_legacy_absolute':error<LEGACY_ATOL,
        'passes_formal_1pct':relative<=.01 if reference else None,
        'passes_pilot_5pct':relative<=.05 if reference else None,
        'passes_zero_reference':observed==0 if reference==0 else None,
        'formal_absolute_error_budget':.01*reference if reference else None}


def rotation_probe(probabilities, policies=None):
    from qiskit import QuantumCircuit,transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_probabilities
    from runtime_identity import apply_backend_options,circuit_fingerprint,normalize
    policies=policies or {'default':{},'low':{'precision':'double','zero_threshold':1e-16,'chop_threshold':1e-16},
        'zero':{'precision':'double','zero_threshold':0.,'chop_threshold':0.}}
    rows=[]
    for p in probabilities:
        if not 0 < p < 1:raise ValueError('rotation target must be in (0,1)')
        a=QuantumCircuit(1);a.ry(2*math.asin(math.sqrt(p)),0)
        compiled=transpile(a,basis_gates=['u','cx'],optimization_level=0,seed_transpiler=917,num_processes=1)
        diagnostic=compiled.copy();diagnostic.save_probabilities([0],label='p')
        for name,extra in policies.items():
            options=dict(method='matrix_product_state',max_parallel_threads=1,mps_omp_threads=1,
                max_parallel_experiments=1,max_parallel_shots=1,matrix_product_state_max_bond_dimension=32,
                matrix_product_state_truncation_threshold=1e-16,seed_simulator=20261030,**extra)
            sim=AerSimulator();backend=apply_backend_options(sim,options)
            start=time.perf_counter();result=sim.run(diagnostic,shots=1).result()
            if not result.success:raise RuntimeError(str(result.status))
            observed=float(result.data()['p'][1])
            rows.append({'target':p,'policy':name,'probability':observed,'seconds':time.perf_counter()-start,
                'comparison':compare_rare_probability(observed,p),'backend':backend,
                'metadata':normalize(result.results[0].metadata),'original_fingerprint':circuit_fingerprint(a),
                'compiled_fingerprint':circuit_fingerprint(compiled),'diagnostic_fingerprint':circuit_fingerprint(diagnostic)})
    return rows


def compiled_sparse_probe(compiled,reference_probabilities,max_states=60000):
    """Retain exact numeric U matrices; do not prune small sparse amplitudes."""
    from qiskit.circuit.library import UnitaryGate
    from reference_engine import evolve,observe
    from runtime_identity import circuit_fingerprint
    circuit=compiled.copy_empty_like();circuit.global_phase=compiled.global_phase
    for ins in compiled.data:
        op=UnitaryGate(ins.operation.to_matrix()) if ins.operation.name=='u' else ins.operation
        circuit.append(op,[circuit.qubits[compiled.find_bit(q).index] for q in ins.qubits])
    row={'compiled_fingerprint':circuit_fingerprint(compiled),'matrix_wrapper_fingerprint':circuit_fingerprint(circuit),
        'conversion':'U gates wrapped with their identical numeric 2x2 matrix; no amplitude pruning',
        'max_states':max_states,'deadline':None}
    start=time.perf_counter()
    try:
        state,runtime=evolve(circuit,max_states=max_states);actual=observe(circuit,state)
        row.update(status='complete',runtime=runtime,observed=actual,
            comparison=compare_rare_probability(actual['probabilities'][1],reference_probabilities[1]))
    except Exception as exc:
        row.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),observed=None,comparison=None)
    row['seconds']=time.perf_counter()-start
    return row


def structure_probe(config_path,steps=2):
    from models import load_model
    from reference_solver import reference
    from unitary_circuit import build_unitary_circuit_qiskit
    from reference_engine import evolve
    from iqae import OracleSpec,build_grover_operator_for_spec
    from runtime_identity import circuit_fingerprint
    cfg=load_model(config_path);a,_=build_unitary_circuit_qiskit(cfg,steps,kernel_profile='compressed')
    state,runtime=evolve(a,max_states=60000);back,back_runtime=evolve(a.inverse(),state,max_states=60000)
    amplitude_error=abs(back.get(0,0)-1);tail=math.fsum(abs(v)**2 for k,v in back.items() if k)
    regs={r.name:r for r in a.qregs};q,obj,_=build_grover_operator_for_spec(A_circuit=a,status_qubits=regs['status'],g_qubits=regs['g'],spec=OracleSpec(good='detect'))
    p=reference(cfg,steps)['rows'][-1]['probabilities'][1]
    psi,_=evolve(q.state_preparation,max_states=60000);rows=[]
    for k in (0,1):
        got=math.fsum(abs(v)**2 for n,v in psi.items() if (n>>obj[0])&1)
        expected=math.sin((2*k+1)*math.asin(math.sqrt(p)))**2
        rows.append({'grover_power':k,**compare_rare_probability(got,expected)})
        if k==0:psi,_=evolve(q,psi,max_states=60000)
    reflected=list(q.reflection_qubits)==list(range(q.num_qubits))
    return {'status':'complete','config':config_path,'steps':steps,'original_A_fingerprint':circuit_fingerprint(a),
        'inverse_fingerprint':circuit_fingerprint(a.inverse()),'grover_fingerprint':circuit_fingerprint(q),
        'inverse_amplitude_error':amplitude_error,'inverse_tail_probability':tail,'runtime':runtime,'inverse_runtime':back_runtime,
        'all_registers_reflected':reflected,'reflection_qubits':list(q.reflection_qubits),'grover_checks':rows,
        'passed':amplitude_error<LEGACY_ATOL and tail<2e-20 and reflected and all(r['passes_formal_1pct'] and r['passes_legacy_absolute'] for r in rows),
        'scope':'Moderate nondegenerate small transport A. Structural check only, not rare-event AE performance.'}


def run_precision_audit(spec,outdir,cpu_budget=1):
    from models import resolve_path
    from runtime_identity import write_json_new,software_versions,source_manifest
    import numpy as np
    output=resolve_path(outdir);output.mkdir(parents=True,exist_ok=True)
    rows=rotation_probe(spec.get('probabilities',[1e-6,1e-8,1e-10,1e-12]),spec.get('policies'))
    structure=structure_probe(spec['structure_config'],spec.get('structure_steps',2)) if spec.get('structure_config') else None
    # Serialize/parse through the same JSON path and assert nonzero is preserved.
    restored=json.loads(json.dumps(rows,allow_nan=False))
    counterexamples=[compare_rare_probability(9e-11,1e-10),compare_rare_probability(0.,1e-10)]
    report={'status':'complete','rows':rows,'structure':structure,'counterexamples':counterexamples,'sources':source_manifest(),
        'versions':software_versions(),'cpu':{'budget':cpu_budget,'workers':1,'threads':1},'deadline':None,
        'float_info':{name:{'eps':float(np.finfo(dtype).eps),'precision_decimal':int(np.finfo(dtype).precision),
            'mantissa_bits':int(np.finfo(dtype).nmant)} for name,dtype in [('float64',np.float64),('longdouble',np.longdouble)]},
        'json_nonzero_preserved':all(r['probability']>0 for r in restored),
        'passed':all(r['comparison']['passes_formal_1pct'] for r in rows) and (structure is None or structure['passed'])}
    write_json_new(output/'precision_audit.json',report)
    lines=['# Rare probability precision audit','',
        'Formal criterion: relative error <=1%; pilot criterion <=5%. Existing absolute 2e-11 is reported separately. No absolute-OR-relative acceptance.',
        'The zero-reference branch requires an exactly zero observed probability and has no relative-error division.',
        'Actual RY gates are transpiled to u/cx and saved through Aer MPS; these are API diagnostics, not transport replacements. Requested/effective options and method metadata are retained.',
        '', '| p target | policy | saved p | relative error | formal 1% |','|---:|---|---:|---:|---|']
    lines += [f"| {r['target']:.3e} | {r['policy']} | {r['probability']:.17e} | {r['comparison']['relative_error']:.3e} | {r['comparison']['passes_formal_1pct']} |" for r in rows]
    lines += ['', 'Counterexamples: 9e-11 versus 1e-10 passes legacy absolute tolerance but fails at 10% relative error; zero versus 1e-10 fails at 100%.',
        'Supported options being accepted/effective is not proof that each threshold acts on every backend operation. Observed policy differences (or lack of differences) are reported without assuming a threshold interpretation.',
        'Float64 and longdouble characteristics are recorded from the installed NumPy; no project-wide precision conversion or arbitrary precision Aer claim.']
    (output/'precision_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    return report


def run_svd_calibration(spec,outdir,cpu_budget=1):
    """Calibrate tiny actual RY/CX circuits; execution and accuracy are distinct.

    The two-qubit state is sqrt(1-p)|00> + sqrt(p)|11>; its small Schmidt
    coefficient exercises SVD, unlike the one-qubit control. These known-p
    circuits are numerical diagnostics and never replace the transport A.
    """
    from qiskit import QuantumCircuit,transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_probabilities
    from models import resolve_path
    from runtime_identity import (apply_backend_options,assert_new_file,circuit_fingerprint,
                                  normalize,source_manifest,software_versions,write_json_new)
    if isinstance(cpu_budget,bool) or not isinstance(cpu_budget,int) or cpu_budget<1:
        raise ValueError('cpu_budget must be a positive integer')
    probabilities=spec.get('probabilities',[1e-6,1e-8,1e-10,1e-12])
    diagnostic_probabilities=spec.get('diagnostic_probabilities',[1e-18])
    for name,values,allow_empty in [('probabilities',probabilities,False),
                                   ('diagnostic_probabilities',diagnostic_probabilities,True)]:
        if not isinstance(values,(list,tuple)) or (not values and not allow_empty):
            raise ValueError(f'{name} must be a nonempty list' if not allow_empty else f'{name} must be a list')
        if any(isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<p<1 for p in values):
            raise ValueError(f'{name} targets must be finite numbers in (0,1)')
    output=resolve_path(outdir)
    json_path=assert_new_file(output/'svd_calibration.json')
    markdown_path=assert_new_file(output/'diagnosis_svd.md')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();rows=[];pairs=[]
    compile_options={'basis_gates':['u','cx'],'optimization_level':0,
                     'seed_transpiler':917,'num_processes':1}
    for scope,targets,truncation in [('formal',probabilities,1e-16),('diagnostic',diagnostic_probabilities,0.)]:
        for target in targets:
            p=float(target)
            for kind,nq in [('ry_1q',1),('entangled_2q',2)]:
                original=QuantumCircuit(nq);original.ry(2*math.asin(math.sqrt(p)),0)
                if nq==2:original.cx(0,1)
                start=time.perf_counter();compiled=transpile(original,**compile_options)
                compile_seconds=time.perf_counter()-start
                event_index=(1<<nq)-1
                analytic=[0.]*(1<<nq);analytic[0]=1-p;analytic[event_index]=p
                start=time.perf_counter()
                sv=[float(x) for x in Statevector.from_instruction(compiled).probabilities()]
                sv_seconds=time.perf_counter()-start
                diagnostic=compiled.copy();diagnostic.save_probabilities(list(range(nq)),label='p')
                fingerprints={'original_fingerprint':circuit_fingerprint(original),
                    'compiled_fingerprint':circuit_fingerprint(compiled),
                    'diagnostic_fingerprint':circuit_fingerprint(diagnostic)}
                pair_rows=[]
                for lapack in (False,True):
                    row={'scope':scope,'circuit_kind':kind,'qubits':nq,'target':p,
                        'event_index':event_index,'basis_order':'little-endian integer index, q0 is least significant',
                        'analytic_probabilities':analytic,'statevector':{'probabilities':sv,'seconds':sv_seconds,
                            'comparison':compare_rare_probability(sv[event_index],p),
                            'normalization_residual':abs(math.fsum(sv)-1.)},
                        'mps_lapack':lapack,'compile_seconds':compile_seconds,
                        'compiled_gate_count':len(compiled.data),**fingerprints}
                    requested=dict(method='matrix_product_state',precision='double',
                        max_parallel_threads=1,mps_omp_threads=1,max_parallel_experiments=1,
                        max_parallel_shots=1,matrix_product_state_max_bond_dimension=2,
                        matrix_product_state_truncation_threshold=truncation,
                        zero_threshold=0.,chop_threshold=0.,mps_lapack=lapack,
                        fusion_enable=False,enable_truncation=False,mps_log_data=True,seed_simulator=20261030)
                    start=time.perf_counter()
                    try:
                        simulator=AerSimulator();row['backend']=apply_backend_options(simulator,requested)
                        result=simulator.run(diagnostic,shots=1).result()
                        if not result.success:raise RuntimeError(str(result.status))
                        # Direct array indexing: no complement subtraction, rounding or renormalization.
                        actual=[float(x) for x in result.data()['p']]
                        if len(actual)!=len(analytic) or any(not math.isfinite(x) or not 0<=x<=1 for x in actual):
                            raise ValueError('invalid saved probability vector')
                        observed=actual[event_index];norm=abs(math.fsum(actual)-1.)
                        error=max(abs(x-y) for x,y in zip(actual,analytic))
                        restored=json.loads(json.dumps(actual,allow_nan=False))
                        row.update(status='complete',probabilities=actual,probability=observed,
                            comparison=compare_rare_probability(observed,p),
                            statevector_max_absolute_error=max(abs(x-y) for x,y in zip(actual,sv)),
                            maximum_absolute_error=error,normalization_residual=norm,
                            passes_legacy_all_probabilities=error<LEGACY_ATOL,
                            passes_normalization=norm<LEGACY_ATOL,
                            json_probability_preserved=restored[event_index]==observed,
                            metadata=normalize(result.results[0].metadata))
                    except Exception as exc:
                        row.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),
                            probability=None,probabilities=None,comparison=None)
                    row['seconds']=time.perf_counter()-start
                    rows.append(row);pair_rows.append(row)
                left,right=pair_rows
                requested_left=left.get('backend',{}).get('requested',{})
                requested_right=right.get('backend',{}).get('requested',{})
                differences=[key for key in set(requested_left)|set(requested_right)
                             if requested_left.get(key)!=requested_right.get(key)]
                pairs.append({'scope':scope,'circuit_kind':kind,'target':p,
                    'only_option_difference_is_mps_lapack':differences==['mps_lapack'],
                    'same_compiled_fingerprint':left['compiled_fingerprint']==right['compiled_fingerprint'],
                    'absolute_probability_difference':abs(left['probability']-right['probability'])
                        if all(r['status']=='complete' for r in pair_rows) else None})
    formal=[r for r in rows if r['scope']=='formal'];diagnostic_rows=[r for r in rows if r['scope']=='diagnostic']
    def accurate(selected,key):
        return all(r['status']=='complete' and r['comparison'][key] for r in selected) if selected else None
    completed=all(r['status']=='complete' for r in rows)
    report={'status':'complete' if completed else 'failed','passed':completed,
        'execution_status':'complete' if completed else 'failed','rows':rows,'lapack_comparisons':pairs,
        'science_status':{'formal_targets_pass_1pct':accurate(formal,'passes_formal_1pct'),
            'formal_targets_pass_5pct':accurate(formal,'passes_pilot_5pct'),
            'diagnostics_pass_1pct':accurate(diagnostic_rows,'passes_formal_1pct'),
            'all_legacy_absolute_pass':all(r.get('passes_legacy_all_probabilities',False) for r in rows),
            'all_normalization_pass':all(r.get('passes_normalization',False) for r in rows)},
        'compile_options':compile_options,'versions':software_versions(),'sources':source_manifest(),
        'cpu':{'budget':cpu_budget,'workers':1,'aer_threads':1,'mps_threads':1},
        'seconds':time.perf_counter()-started,'deadline':None,'counts_check':'skipped; direct save calibration only',
        'scope':'Tiny numerical calibration; formal rows target 1e-6 through 1e-12 by default. '
            'The 1e-18/truncation-zero control is diagnostic only, not a transport or AE result.',
        'source_evidence':{'version':'0.17.2','installed_version_matches':software_versions()['qiskit-aer']=='0.17.2',
            'header':'https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/simulators/matrix_product_state/svd.hpp#L23',
            'implementation':'https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/simulators/matrix_product_state/svd.cpp#L76-L130',
            'finding':'Version source defines CHOP_THRESHOLD=1e-16; reduce_zeros counts singular values whose squared magnitude exceeds it before runtime truncation.',
            'limitation':'Version-tag source evidence plus installed option/behavior checks; installed binary was not rebuilt or instruction-traced. Does not identify the sole cause of tau23 zero.'}}
    report=normalize(report);write_json_new(json_path,report)
    lines=['# SVD / 稀有概率链路小诊断','',
        f"执行状态：{report['execution_status']}；正式目标 1%：{report['science_status']['formal_targets_pass_1pct']}；额外诊断 1%：{report['science_status']['diagnostics_pass_1pct']}。",
        '1q: RY(2 asin(sqrt(p)))；2q: 同一 RY 后接 CX，目标为 |11>。解析值和编译电路的双精度 Statevector 独立对照；没有使用 initialize 或已知输运输出。',
        'MPS 直接保存全部基态概率，取整数索引 1 或 3。没有补数相减、格式化置零或重归一化；JSON 保留原浮点值。计数未执行。',
        '每对仅 mps_lapack 改变。bond=2 足够表示二量子位；double、zero/chop=0、单线程；fusion 与 qubit truncation 关闭，保存完整 MPS metadata。正式运行截断为 1e-16，额外 p=1e-18 诊断截断为 0。','',
        '| scope | circuit | p | LAPACK | saved p | relative error | norm residual | seconds |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        if r['status']=='complete':
            lines.append(f"| {r['scope']} | {r['circuit_kind']} | {r['target']:.1e} | {r['mps_lapack']} | {r['probability']:.17e} | {r['comparison']['relative_error']:.4e} | {r['normalization_residual']:.3e} | {r['seconds']:.4f} |")
        else:lines.append(f"| {r['scope']} | {r['circuit_kind']} | {r['target']:.1e} | {r['mps_lapack']} | failed | — | — | {r['seconds']:.4f} |")
    lines += ['',
        '源码证据：[Aer 0.17.2 svd.hpp](https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/simulators/matrix_product_state/svd.hpp#L23) 定义内部 CHOP_THRESHOLD=1e-16；[svd.cpp](https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/simulators/matrix_product_state/svd.cpp#L76-L130) 在运行截断前按奇异值平方筛除。运行 chop_threshold=0 不保证所有内部筛除取消。已核对安装版本与 requested/effective 选项；未重编译或逐指令跟踪安装二进制。',
        '每行都保留解析/Statevector/MPS 三种值、原始/编译/带 save 电路指纹、耗时与规范残差。passed 仅表示诊断执行完整；science_status 单列相对 1%、5%、旧绝对 2e-11 与归一性，不用绝对误差抵消稀有概率失败。',
        '本诊断不能把 τ23 的归零唯一归因于内部阈值：完整 A 还涉及很多 Schmidt 分量、bond 上限和累积数值误差。小例通过也不能外推深电路的相对精度。',
        '下一项云端应只变一个因素。若两条 LAPACK 路径在本小例没有明显差异，结合既有 τ23/bond128 触顶记录，优先建议 bond128→256，保持模型、T15、编译、运行截断1e-16、LAPACK=false、线程、seed 不变；关闭重复 counts 仅改变测量诊断成本，不改变保存概率的 A。该建议不是恢复概率的保证，云端由用户手动启动；LAPACK 切换可作为另一独立对照，不能同时改变。']
    with markdown_path.open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
    return report
