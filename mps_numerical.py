"""Finite numerical diagnostics, using independent unnormalized contractions.

All save operations are diagnostic only. No saved tensor, reference probability,
or normalization correction is fed into evolution or a statistical estimator.
"""
import math
import time
import traceback
from dataclasses import asdict


def mps_norm(state):
    """Contract Aer Gamma/Lambda tensors without canonicality assumptions.

    E <- sum_s B_s.conj().T @ E @ B_s, B_i = Gamma_i Lambda_i.
    This is ordinary floating arithmetic independent of Aer's probability-save
    code, not an interval bound and not an exact-reference evolution.
    """
    import numpy as np
    gammas,lambdas=state
    if not gammas or len(lambdas)!=len(gammas)-1:
        raise ValueError('Invalid Gamma/Lambda tensor count')
    environment=np.ones((1,1),dtype=complex)
    for i,pair in enumerate(gammas):
        if len(pair)!=2:raise ValueError('Expected two physical Gamma matrices')
        matrices=[np.asarray(g,dtype=complex) for g in pair]
        if any(g.ndim!=2 or g.shape[0]!=environment.shape[0] for g in matrices):
            raise ValueError('Invalid MPS bond shape')
        if i<len(lambdas):
            lam=np.asarray(lambdas[i],dtype=float)
            if any(g.shape[1]!=len(lam) for g in matrices):raise ValueError('Invalid Lambda shape')
            matrices=[g*lam[None,:] for g in matrices]
        environment=sum(g.conj().T@environment@g for g in matrices)
    if environment.shape!=(1,1):raise ValueError('MPS has open external bond')
    value=complex(environment[0,0])
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError('Nonfinite MPS norm contraction')
    if abs(value.imag)>2e-11:raise ValueError('MPS norm contraction has nonreal residual')
    return value.real


def append_readout_before(circuit):
    from qiskit_aer.library import save_matrix_product_state
    circuit.save_matrix_product_state(label='audit_mps_before')


def append_readout_after(circuit,status_bits,*,include_mps=True):
    from qiskit_aer.library import save_density_matrix,save_matrix_product_state,save_probabilities
    circuit.save_density_matrix(status_bits,label='audit_status_density',unnormalized=True)
    circuit.save_probabilities(status_bits,label='audit_status_after')
    if include_mps:circuit.save_matrix_product_state(label='audit_mps_after')


def readout_record(data,probabilities,*,include_mps=True):
    import numpy as np
    from validation import TOL
    start=time.perf_counter()
    before=mps_norm(data['audit_mps_before']) if include_mps else None
    after=mps_norm(data['audit_mps_after']) if include_mps else None
    density=np.asarray(data['audit_status_density'])
    diag=np.diag(density);saved_after=np.asarray(data['audit_status_after'])
    density_error=max(abs(complex(a)-b) for a,b in zip(diag,probabilities))
    repeat_error=max(abs(a-b) for a,b in zip(saved_after,probabilities))
    norm=math.fsum(probabilities)
    norm_error=max(abs(norm-1.),abs(complex(np.trace(density))-1.))
    consistency=max(density_error,repeat_error)
    if include_mps:
        consistency=max(consistency,abs(before-norm),abs(after-norm),abs(after-before))
        norm_error=max(norm_error,abs(before-1.),abs(after-1.))
    return {'status_density_diagonal_real':diag.real.tolist(),
        'status_density_diagonal_imag':diag.imag.tolist(),
        'status_probability_after_density':saved_after.tolist(),
        'tensor_norm_before':before,'tensor_norm_after':after,
        'tensor_norm_before_residual':abs(before-1.) if include_mps else None,
        'tensor_norm_after_residual':abs(after-1.) if include_mps else None,
        'tensor_norm_before_after_abs':abs(before-after) if include_mps else None,
        'saved_probability_norm':norm,'density_trace_real':float(np.trace(density).real),
        'density_trace_imag':float(np.trace(density).imag),
        'density_vs_saved_max_abs':float(density_error),'repeated_probability_max_abs':float(repeat_error),
        'tensor_vs_saved_norm_max_abs':max(abs(before-norm),abs(after-norm)) if include_mps else None,
        'consistency_status':'passed' if consistency<TOL else 'failed',
        'normalization_status':'passed' if norm_error<TOL else 'failed',
        'tensor_snapshot_enabled':include_mps,
        'absolute_tolerance':TOL,'contraction_seconds':time.perf_counter()-start,
        'scope':('Independent floating tensor contraction and two Aer marginal readouts within the instrumented run. Full MPS saves can activate otherwise unused qubits and perturb floating tensor-canonicalization paths even when the active map is unchanged; this instrumentation must not be equated to ordinary marginal-only evolution. No normalization, clipping, or probability correction. Saved tensors are transient.' if include_mps else
                 'Two Aer marginal readouts on the same evolved state, with no full-state/MPS snapshot. Check active-input mapping before attributing comparisons. No independent tensor contraction or normalization correction.')}


def run_numerical_audit(spec,outdir,cpu_budget=1):
    """One small full-step transport case, fixed compiled A, LAPACK 0/1 only."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_probabilities
    from models import load_model
    from unitary_circuit import build_unitary_circuit_qiskit
    from reference_engine import evolve,observe,registers,marginal
    from reference_solver import reference
    from rare_precision import evolve_with_pruning_bound,event_pruning_bound
    from runtime_identity import circuit_fingerprint,write_json_new,apply_backend_options,normalize
    from numerical_runtime import configure_blas_threads,numerical_libraries
    from validation import (_output,_validated_cases,_probe_settings,_probe_backend_options,
                            _identity,_probe_accuracy_status,_probe_summary_row,TOL)
    cases=_validated_cases(spec);bonds,thresholds,_=_probe_settings(spec)
    if len(cases)!=1 or len(bonds)!=1 or len(thresholds)!=1 or spec.get('counts_check') is not False:
        raise ValueError('numerical audit requires one case/bond/threshold, counts_check=false')
    case=cases[0];profile=case.get('profile','baseline');steps=case['steps']
    if steps>3:raise ValueError('Local numerical audit is limited to at most three physical steps')
    cfg=load_model(case['config']);output=_output(outdir)
    original,_=build_unitary_circuit_qiskit(cfg,steps,kernel_profile=profile)
    compiled=transpile(original,basis_gates=['u','cx'],optimization_level=0,seed_transpiler=917,num_processes=1)
    ref=reference(cfg,steps,retain_joint=True)
    target=ref['rows'][-1]['probabilities'];budget=int(spec.get('max_sparse_states',60000))
    max_states=budget
    state,raw_runtime=evolve(original,max_states=max_states)
    raw=observe(original,state);raw_joint=marginal(original,state,['status','x','y','z','dir','g'])
    expected_joint=ref['joint']
    joint_error=max(abs(raw_joint.get(k,0)-expected_joint.get(k,0)) for k in set(raw_joint)|set(expected_joint))
    raw_fp=circuit_fingerprint(original);compiled_fp=circuit_fingerprint(compiled)
    exact_compiled={'status':'not_evaluated'}
    try:
        exact_state,runtime=evolve(compiled,max_states=max_states)
        exact_compiled={'status':'complete','observables':observe(compiled,exact_state),'runtime':runtime}
        exact_observed=exact_compiled['observables']
        exact_compiled['max_status_abs_error']=max(abs(a-b) for a,b in zip(exact_observed['probabilities'],raw['probabilities']))
        exact_compiled['passed']=max(exact_compiled['max_status_abs_error'],
            abs(exact_observed['norm']-1.),exact_observed['work_nonzero'])<TOL
    except MemoryError as exc:
        exact_compiled={'status':'state_budget_exceeded','error':str(exc),'max_states':max_states,
                        'scope':'No cutoff and no normalization; failure retained, not counted as exact certification.'}
    pruned,runtime=evolve_with_pruning_bound(compiled,float(spec.get('amplitude_cutoff',1e-15)),max_states)
    bounded=observe(compiled,pruned);delta=runtime['discarded_l2_sum']
    errors=[abs(a-b) for a,b in zip(raw['probabilities'],bounded['probabilities'])]
    bounds=[event_pruning_bound(p,delta) for p in bounded['probabilities']]
    bounded.update(runtime=runtime,status_abs_errors=errors,pruning_only_error_bounds=bounds,
                   status_error_plus_pruning_bound=[a+b for a,b in zip(errors,bounds)],
                   floating_roundoff_certified=False)
    bounded['work_error_plus_pruning_bound']=bounded['work_nonzero']+event_pruning_bound(bounded['work_nonzero'],delta)
    bounded['norm_error_plus_pruning_bound']=abs(bounded['norm']-1.)+2*math.sqrt(bounded['norm'])*delta+delta**2
    regs=registers(compiled);diagnostic=compiled.copy()
    ordinary=compiled.copy()
    ordinary.save_probabilities(regs['status'],label='status_probability')
    for i,bit in enumerate(regs['work']):ordinary.save_probabilities([bit],label=f'work_{i}')
    density_only=ordinary.copy();append_readout_after(density_only,regs['status'],include_mps=False)
    append_readout_before(diagnostic)
    diagnostic.save_probabilities(regs['status'],label='status_probability')
    for i,bit in enumerate(regs['work']):diagnostic.save_probabilities([bit],label=f'work_{i}')
    append_readout_after(diagnostic,regs['status'])
    rows=[]
    for lapack in (False,True):
        extras=dict(spec.get('backend_options',{}));extras['mps_lapack']=lapack
        options=_probe_backend_options(cpu_budget,1,1,extras,bonds[0],thresholds[0],spec.get('seed',20261030))
        sim=AerSimulator();backend=apply_backend_options(sim,options)
        libraries_before=numerical_libraries();application=configure_blas_threads(1)
        marginal_variants=[]
        for name,observed_circuit in [('ordinary',ordinary),('density_only',density_only)]:
            tick=time.perf_counter();cpu_tick=time.process_time()
            measured=sim.run(observed_circuit,shots=1).result()
            seconds=time.perf_counter()-tick;cpu_seconds=time.process_time()-cpu_tick
            if not measured.success:raise RuntimeError(measured.status)
            values=measured.data();p=values['status_probability'].tolist()
            w=[float(values[f'work_{i}'][1]) for i in range(len(regs['work']))]
            variant={'readout_variant':name,'probabilities':p,'reference':target,
                'absolute_errors':[abs(a-b) for a,b in zip(p,target)],
                'normalization_residual':abs(math.fsum(p)-1.),'work_bit_probabilities':w,
                'work_any_nonzero_union_upper_bound':math.fsum(w),
                'identity':_identity(cfg,steps,profile,original,backend,'numerical_readout_audit',raw_fp,
                                    compiled_fp,circuit_fingerprint(observed_circuit)),
                'metadata':normalize(measured.results[0].metadata),
                'mps_log_scope':'process-cumulative; no per-setting bond attribution',
                'evolution_and_read_wall_seconds':seconds,'process_cpu_seconds':cpu_seconds,
                'process_average_cores':cpu_seconds/seconds,
                **_probe_accuracy_status(p,target,math.fsum(w),work_bit_probabilities=w)}
            if name=='density_only':variant['readout_diagnostics']=readout_record(values,p,include_mps=False)
            variant_path=write_json_new(output/(name+'_lapack_'+str(int(lapack))+'.json'),variant)
            marginal_variants.append(_probe_summary_row(variant,variant_path))
        started=time.perf_counter();cpu_start=time.process_time()
        result=sim.run(diagnostic,shots=1).result()
        elapsed=time.perf_counter()-started;cpu=time.process_time()-cpu_start
        if not result.success:raise RuntimeError(result.status)
        data=result.data();probs=data['status_probability'].tolist()
        work=[float(data[f'work_{i}'][1]) for i in range(len(regs['work']))]
        row={'case':case,'config':asdict(cfg),'profile':profile,'steps':steps,
             'original_A_fingerprint':raw_fp,'compiled_A_fingerprint':compiled_fp,
             'diagnostic_fingerprint':circuit_fingerprint(diagnostic),'checkpoint':'complete physical step; all uncomputation complete',
             'reference':target,'probabilities':probs,'absolute_errors':[abs(a-b) for a,b in zip(probs,target)],
             'normalization_residual':abs(math.fsum(probs)-1.),'work_bit_probabilities':work,
             'work_any_nonzero_union_upper_bound':math.fsum(work),'readout_diagnostics':readout_record(data,probs),
             'backend':backend,'blas_application':application,'numerical_libraries_before':libraries_before,
             'numerical_libraries_after':numerical_libraries(),'evolution_and_read_wall_seconds':elapsed,
             'process_cpu_seconds':cpu,'process_average_cores':cpu/elapsed,
             'identity':_identity(cfg,steps,profile,original,backend,'numerical_readout_audit',raw_fp,
                                  compiled_fp,circuit_fingerprint(diagnostic)),
             'metadata':normalize(result.results[0].metadata),
             'mps_log_scope':'Aer process-cumulative metadata; no per-setting bond attribution from this log',
             **_probe_accuracy_status(probs,target,math.fsum(work),work_bit_probabilities=work)}
        ordinary_row,density_row=marginal_variants
        row['ordinary_marginal_baseline']=ordinary_row;row['density_only_variant']=density_row
        mapping_keys=('active_input_qubits','input_qubit_map','num_qubits')
        row['readout_variant_comparison']={
            'density_vs_ordinary_status_max_abs':max(abs(a-b) for a,b in zip(density_row['probabilities'],ordinary_row['probabilities'])),
            'tensor_save_vs_ordinary_status_max_abs':max(abs(a-b) for a,b in zip(probs,ordinary_row['probabilities'])),
            'density_same_active_mapping':all(density_row['metadata'][k]==ordinary_row['metadata'][k] for k in mapping_keys),
            'tensor_save_same_active_mapping':all(row['metadata'][k]==ordinary_row['metadata'][k] for k in mapping_keys),
            'scope':'Only readout instrumentation changes within each LAPACK setting. Tensor saves may change backend representation/canonicalization; differences are diagnostic, not correction factors.'}
        path=write_json_new(output/('lapack_'+str(int(lapack))+'.json'),row)
        rows.append(_probe_summary_row(row,path))
    error=max(abs(a-b) for a,b in zip(rows[0]['probabilities'],rows[1]['probabilities']))
    report={'kind':'numerical_readout_audit','case':case,'raw_independent':raw,'raw_runtime':raw_runtime,
            'raw_independent_joint_max_error':joint_error,'compiled_unpruned':exact_compiled,
            'compiled_bounded':bounded,'rows':rows,'lapack_status_max_abs_difference':error,
            'passed':joint_error<TOL and raw['work_nonzero']<TOL and abs(raw['norm']-1.)<TOL and
                max(a+b for a,b in zip(errors,bounds))<TOL and error<TOL and
                bounded['work_error_plus_pruning_bound']<TOL and bounded['norm_error_plus_pruning_bound']<TOL and
                exact_compiled.get('passed',True) and
                all(r['strict_status']=='passed' and r['work_accuracy_status']=='passed' and
                    r['readout_diagnostics']['consistency_status']=='passed' and
                    r['readout_diagnostics']['normalization_status']=='passed' and
                    r['ordinary_marginal_baseline']['strict_status']=='passed' and
                    r['ordinary_marginal_baseline']['work_accuracy_status']=='passed' and
                    r['density_only_variant']['strict_status']=='passed' and
                    r['density_only_variant']['work_accuracy_status']=='passed' and
                    r['density_only_variant']['readout_diagnostics']['consistency_status']=='passed' and
                    r['density_only_variant']['readout_diagnostics']['normalization_status']=='passed' and
                    r['readout_variant_comparison']['density_same_active_mapping'] and
                    r['readout_variant_comparison']['density_vs_ordinary_status_max_abs']<TOL
                    for r in rows),
            'deadline':None,'scope':'Small full-step transport diagnostic. State-budget failure is explicit; bounded compiled comparison excludes certification of floating arithmetic. This does not certify the full T15 main candidate.'}
    write_json_new(output/'numerical_audit.json',report)
    return report
