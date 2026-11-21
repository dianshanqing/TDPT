"""Worker-side AE execution. Public callers use iqae's supervised wrappers."""
from __future__ import annotations
import hashlib,math,time
from ae_runtime import budget,BudgetExceeded
from ae_likelihood import integer
from iqae_diagnostics import record_stage


def run_sampler(sampler,pubs):
    import iqae
    from supervised import publish_progress
    sampler._call_id+=1;call=sampler._call_id
    run_budget=getattr(sampler,'_budget',None) or budget(timeout=sampler._max_call_seconds)
    sampler._budget=run_budget;items=[];completed_call=0
    if not hasattr(sampler,'_used_seeds'):sampler._used_seeds=set()
    for pub_id,pub in enumerate(pubs):
        qc=pub[0] if isinstance(pub,tuple) else pub
        power=integer((getattr(qc,'metadata',None) or {}).get('grover_power',getattr(sampler,'active_power',0) or 0),'power')
        run_budget.check('before_compile',power)
        attempt={'call_id':call,'pub_id':pub_id,'power':power,'shots':sampler._shots}
        progress={'scope':'sampler_pub','call_id':call,'pub_id':pub_id,'power':power}
        run_budget.record('compile_start',**attempt)
        publish_progress('ae_compile',shots_completed=0,shots_total=sampler._shots,**progress)
        compile_stage='measurement_wrap'
        try:
            with record_stage(run_budget.record,compile_stage,**attempt):
                measured=sampler._force_measure_objective_only(qc)
            compile_stage='transpile'
            with record_stage(run_budget.record,compile_stage,**attempt):
                compiled=sampler._compile_for_aer(measured,call_id=call,pub_id=pub_id)
            del measured
            from runtime_identity import circuit_fingerprint
            compile_stage='fingerprint'
            with record_stage(run_budget.record,compile_stage,**attempt):
                fingerprint=circuit_fingerprint(compiled)
            compile_stage='circuit_stats'
            with record_stage(run_budget.record,compile_stage,**attempt):
                depth=compiled.depth();gate_count={str(k):int(v) for k,v in compiled.count_ops().items()}
            run_budget.record('compile_complete',**attempt,circuit_fingerprint=fingerprint,
                circuit_depth=depth,gate_count=gate_count,
                measurement_mapping={'objective_index':sampler._obj,'classical_register':'cobj','classical_width':1})
            publish_progress('ae_compile_complete',shots_completed=0,shots_total=sampler._shots,**progress)
        except Exception as exc:
            run_budget.record('compile_error',**attempt,error=repr(exc),failed_stage=compile_stage)
            publish_progress('ae_compile_error',shots_completed=0,shots_total=sampler._shots,error=repr(exc),**progress)
            raise
        counts={};done=0;chunk=0
        while done<sampler._shots:
            run_budget.check('before_submit',power)
            shots=min(sampler._shot_chunk or sampler._shots,sampler._shots-done)
            token=f'{sampler._seed}:{call}:{pub_id}:{chunk}'.encode()
            seed=int.from_bytes(hashlib.sha256(token).digest()[:4],'little')
            if seed in sampler._used_seeds:raise RuntimeError('derived random stream collision')
            sampler._used_seeds.add(seed)
            details={'call_id':call,'pub_id':pub_id,'chunk_id':chunk,'power':power,'shots':shots,'seed_simulator':seed,
                'sample_start':done,'sample_stop':done+shots}
            run_budget.record('sample_submit',**details)
            publish_progress('ae_submit',shots_completed=done,shots_total=sampler._shots,
                shots_inflight=shots,chunk_id=chunk,**progress)
            job=None;started=time.monotonic()
            try:
                with record_stage(run_budget.record,'backend_submit',**details):
                    job=sampler._backend.run(compiled,shots=shots,seed_simulator=seed)
                while not job.done():
                    run_budget.check('sampling',power);time.sleep(.01)
                run_budget.check('before_result',power)
                part=job.result().get_counts()
                if any(type(v) is not int or v<0 for v in part.values()) or sum(part.values())!=shots:
                    raise RuntimeError('backend raw counts do not match requested shots')
                normalized={'0':0,'1':0}
                for key,n in part.items():
                    key=str(key).replace(' ','')
                    if key not in ('0','1'):raise RuntimeError('objective measurement is not one bit')
                    normalized[key]+=n;counts[key]=counts.get(key,0)+n
                run_budget.record('sample_complete',**details,counts=normalized,sampling_seconds=time.monotonic()-started)
                done+=shots;chunk+=1
                publish_progress('ae_chunk_complete',shots_completed=done,shots_total=sampler._shots,
                    shots_inflight=0,chunk_id=chunk-1,**progress)
                if sampler._progress_state is not None:sampler._progress_state.update(stage='sampler',shots_done=done,shots_total=sampler._shots)
            except Exception as exc:
                cancel={'requested':False,'backend_confirmed':False,'error':None}
                if job is not None:
                    cancel['requested']=True
                    try:cancel['return_value']=bool(job.cancel());cancel['backend_confirmed']=bool(job.done())
                    except Exception as ce:cancel['error']=repr(ce)
                run_budget.record('sample_error',**details,error=repr(exc),status=getattr(exc,'status','failed'),
                    cancellation=cancel,sampling_seconds=time.monotonic()-started,completed_counts_before_error=counts.copy(),
                    hard_cleanup='supervised worker process boundary; see supervision.all_processes_reaped')
                publish_progress('ae_sample_error',shots_completed=done,shots_total=sampler._shots,
                    shots_inflight=None,chunk_id=chunk,error=repr(exc),**progress)
                raise
        data=iqae._SamplerData('meas',iqae._CountsLike(counts))
        items.append(iqae._SamplerResultItem(data,{'shots':sampler._shots}))
        completed_call+=done
        publish_progress('ae_pub_complete',shots_completed=done,shots_total=sampler._shots,**progress)
    publish_progress('ae_sampler_complete',shots_completed=completed_call,shots_total=completed_call,
        scope='sampler_call',call_id=call)
    return iqae._JobLike(items)


def _limits(kwargs):
    cap=kwargs.get('iqae_max_grover_power')
    cap=16 if cap is None else integer(cap,'iqae_max_grover_power')
    if cap<0:raise ValueError('power cap must be >=0')
    timeout=kwargs.get('sampler_max_call_seconds')
    timeout=None if timeout is None else float(timeout)
    return budget(timeout=timeout,max_power=cap),cap


def _sampler(q,objective,kwargs,run_budget):
    import iqae
    sampler=iqae._PubsToAerObjectiveSampler(shots=integer(kwargs.get('shots',2000),'shots'),objective_qubit_index=objective,
        seed=kwargs.get('seed'),verbose=bool(kwargs.get('verbose_timing',True)),backend_options=kwargs.get('backend_options'),
        debug=bool(kwargs.get('debug_sampler',False)),heartbeat_sec=kwargs.get('sampler_heartbeat_sec',30),
        shot_chunk=kwargs.get('sampler_shot_chunk',100),print_ops=kwargs.get('sampler_print_ops',False),
        max_call_seconds=kwargs.get('sampler_max_call_seconds'),progress_state=kwargs.get('progress_state'))
    sampler._budget=run_budget;sampler.active_power=0
    run_budget.record('backend_configured',backend_settings=sampler.effective_backend_options)
    return sampler


def _start(A,status,g,spec,kwargs):
    import iqae
    from runtime_identity import circuit_fingerprint
    b,cap=_limits(kwargs);b.check('build_grover_start')
    with record_stage(b.record,'build_grover',max_power=cap):
        Q,objective,marker=iqae.build_grover_operator_for_spec(A_circuit=A,status_qubits=status,g_qubits=g,spec=spec)
    b.check('build_grover_complete')
    sampler=_sampler(Q,objective[0],kwargs,b)
    with record_stage(b.record,'fingerprint',component='marked_A'):
        state_preparation_fingerprint=circuit_fingerprint(Q.state_preparation)
    with record_stage(b.record,'fingerprint',component='Q'):
        grover_fingerprint=circuit_fingerprint(Q)
    with record_stage(b.record,'circuit_stats',component='grover_components'):
        state_stats=iqae._circuit_stats(Q.state_preparation);grover_stats=iqae._circuit_stats(Q)
    meta={'status':'running','good':spec.good,'target_g':spec.target_g,'rare_eta':spec.rare_eta,'p_target':spec.p_target,
        'objective_qubits':objective,'measurement_mapping':{'objective_indices':objective,'classical_register':'cobj','width':1},
        'marker_meta':marker,'confidence_interval':None,'backend_settings':getattr(sampler,'effective_backend_options',None),
        'state_preparation_fingerprint':state_preparation_fingerprint,
        'grover_fingerprint':grover_fingerprint,'reflection_qubits':list(Q.reflection_qubits),
        'all_qubits_reflected':list(Q.reflection_qubits)==list(range(Q.num_qubits)),
        'state_preparation_stats':state_stats,'grover_operator_stats':grover_stats,
        'iqae_max_grover_power':cap,'capped_iqae_enforced':True,'automatic_retries':0}
    return Q,objective,sampler,b,meta


def run_iqae(*,A_circuit,status_qubits,g_qubits,spec,**kwargs):
    import iqae
    if kwargs.get('use_aer_sampler',True) is not True:raise ValueError('only the supervised Aer objective sampler is supported')
    Q,obj,sampler,b,meta=_start(A_circuit,status_qubits,g_qubits,spec,kwargs)
    # Explicit legacy epsilon calls retain their absolute meaning. The public
    # production entry passes precision_mode='relative' unconditionally.
    mode=kwargs.get('precision_mode','absolute' if 'epsilon_target' in kwargs else 'relative')
    if mode not in ('relative','absolute'):raise ValueError('unknown precision_mode')
    if mode=='relative':
        def sample_counts(k,n):
            b.check('power_selection',k)
            sampler._shots=n; sampler.active_power=k
            with record_stage(b.record,'power_build',power=k):
                circuit=iqae._build_power_circuit(Q,k)
                circuit.metadata={**(circuit.metadata or {}),'grover_power':k}
            counts=sampler.run([(circuit,)]).result()[0].data.meas.get_counts()
            return counts.get('1',0)
        result=iqae.estimate_relative_iqae(sample_counts,
            relative_error=kwargs.get('relative_error',.01),confidence=kwargs.get('confidence',.95),
            shots=kwargs.get('shots',512),max_shots=kwargs.get('max_shots',200000),
            max_queries=kwargs.get('max_queries',100000000000),max_power=meta['iqae_max_grover_power'])
        meta.update(result)
        meta.update(precision_mode='relative',num_oracle_queries=result['costs']['grover_iterations'],
                    statistical_costs=result['costs'],powers=[r['power'] for r in result['rounds']],
                    costs=b.summary(),attempts=b.events,timing_sec={'total':time.monotonic()-b.started})
        return result['estimate'],meta
    IAE,Problem=iqae._import_qiskit_algorithms()
    epsilon=float(kwargs.get('epsilon_target',.05));alpha=float(kwargs.get('alpha',.05))
    if not math.isfinite(epsilon) or not 0<epsilon<.5 or not math.isfinite(alpha) or not 0<alpha<1:raise ValueError('invalid IQAE epsilon/alpha')
    problem=Problem(state_preparation=Q.state_preparation,grover_operator=Q,objective_qubits=obj)
    algorithm=IAE(epsilon_target=epsilon,alpha=alpha);algorithm._sampler=sampler
    original_construct=algorithm.construct_circuit
    def checked_construct(problem,k=0,measurement=False):
        b.check('power_selection',k)  # before even Q.power(k) is built
        sampler.active_power=int(k)
        with record_stage(b.record,'power_build',power=int(k)):
            circuit=original_construct(problem,k,measurement)
            circuit.metadata={**(circuit.metadata or {}),'grover_power':int(k)}
        b.check('power_build_complete',k)
        return circuit
    algorithm.construct_circuit=checked_construct
    try:
        result=algorithm.estimate(problem)
        b.check('estimate_complete')
        meta.update(status='completed',epsilon_target=epsilon,alpha=alpha,confidence_interval=list(map(float,result.confidence_interval)),
            num_oracle_queries=int(result.num_oracle_queries),library_query_unit='Grover Q iterations',
            powers=list(map(int,getattr(result,'powers',[]))),precision_target_met=True,
            estimate_intervals=getattr(result,'estimate_intervals',None),theta_intervals=getattr(result,'theta_intervals',None))
        estimate=float(result.estimation)
    except Exception as exc:
        cause=exc
        while cause.__cause__ is not None:cause=cause.__cause__
        status=getattr(cause,'status','failed')
        meta.update(status=status,error=repr(exc),root_error=repr(cause),precision_target_met=False,confidence_interval=None)
        b.record('estimate_incomplete',status=status,error=repr(exc));estimate=None
    meta.update(costs=b.summary(),attempts=b.events,timing_sec={'total':time.monotonic()-b.started})
    return estimate,meta


def run_fixed(*,A_circuit,status_qubits,g_qubits,spec,powers,**kwargs):
    import iqae
    powers=[integer(k,'power') for k in powers]
    if not powers or any(k<0 for k in powers):raise ValueError('nonempty nonnegative powers required')
    if len(set(powers))!=len(powers):raise ValueError('duplicate powers require explicit independent-run grouping, not repeated seed reuse')
    b,cap=_limits(kwargs)
    # Reject the full requested schedule before building its first power.
    try:
        for k in powers:b.check('schedule_validation',k)
    except BudgetExceeded as exc:
        return {'rows':[],'meta':{'status':exc.status,'error':str(exc),'precision_target_met':False,'backend_settings':None,'attempts':b.events,'costs':b.summary()}}
    Q,obj,sampler,b,meta=_start(A_circuit,status_qubits,g_qubits,spec,kwargs)
    return sample_schedule(Q,sampler,powers,b,meta)


def sample_schedule(Q,sampler,powers,b,meta):
    import iqae
    rows=[]
    try:
        for k in powers:
            b.check('power_build_start',k);started=time.monotonic()
            with record_stage(b.record,'power_build',power=k):
                circuit=iqae._build_power_circuit(Q,k)
                circuit.metadata={**(circuit.metadata or {}),'grover_power':k};sampler.active_power=k
            build_seconds=time.monotonic()-started;b.check('power_build_complete',k)
            sample_started=time.monotonic()
            counts=sampler.run([(circuit,)]).result()[0].data.meas.get_counts();n=sum(counts.values());h=counts.get('1',0)
            rows.append({'m':k,'shots':n,'success_count':h,'failure_count':n-h,'oracle_calls':2*k+1,
                'success_probability':h/n,'success_se':math.sqrt((h/n)*(1-h/n)/n),
                'build_power_time_sec':build_seconds,'build_time_sec':build_seconds,'elapsed_time_sec':time.monotonic()-sample_started,
                'circuit_depth':circuit.depth(),'qubit_count':circuit.num_qubits,
                'gate_count':{str(key):int(value) for key,value in circuit.count_ops().items()}})
        meta['status']='completed'
    except Exception as exc:
        meta.update(status=getattr(exc,'status','failed'),error=repr(exc));b.record('fixed_incomplete',status=meta['status'],error=repr(exc))
    meta.update(costs=b.summary(),attempts=b.events,fit_status='not_fit',precision_target_met=False,timing_sec={'total':time.monotonic()-b.started})
    return {'rows':rows,'meta':meta}
