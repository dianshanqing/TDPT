"""Finite diagnostics of the production IQAE circuits, separate from estimation.

Exports preserve exact gate identities. Probability saves and sparse evolution
are observations only: no result here is an IQAE estimate or stopping input.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import ctypes
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

from runtime_identity import (assert_new_file, circuit_fingerprint, normalize,
    software_versions, source_manifest, validate_output_path, write_json_new)

QPY_READER='qiskit.qpy.binary_io.read_circuit(use_rust=False); checked single-circuit header'


def resource_snapshot():
    """Endpoint counters; peaks are OS lifetime high-water marks, not RSS caps."""
    result=dict(rss_bytes=None, process_peak_rss_bytes=None,
        process_commit_bytes=None, process_peak_commit_bytes=None,
        system_commit_bytes=None, system_commit_limit_bytes=None,
        system_peak_commit_bytes=None, physical_available_bytes=None,
        job_commit_bytes=None, job_peak_commit_bytes=None, job_commit_limit_bytes=None,
        peak_scope='process/job lifetime; system since boot', errors=[])
    try:
        import psutil
        info=psutil.Process().memory_info()
        result.update(rss_bytes=int(info.rss),
            process_peak_rss_bytes=getattr(info,'peak_wset',None),
            physical_available_bytes=int(psutil.virtual_memory().available))
        if os.name=='nt':
            result.update(process_commit_bytes=int(info.pagefile),
                          process_peak_commit_bytes=int(info.peak_pagefile))
    except Exception as exc:result['errors'].append('process_memory: '+repr(exc))
    if os.name!='nt':return result
    from ctypes import wintypes as w
    try:
        # Microsoft PERFORMANCE_INFORMATION: commit counters are pages.
        class Performance(ctypes.Structure):
            _fields_=[('cb',w.DWORD)]+[(n,ctypes.c_size_t) for n in (
                'CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
                'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[
                (n,w.DWORD) for n in ('HandleCount','ProcessCount','ThreadCount')]
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetPerformanceInfo.argtypes=[ctypes.POINTER(Performance),w.DWORD]
        psapi.GetPerformanceInfo.restype=w.BOOL
        data=Performance();data.cb=ctypes.sizeof(data)
        if not psapi.GetPerformanceInfo(ctypes.byref(data),data.cb):raise ctypes.WinError(ctypes.get_last_error())
        result.update(system_commit_bytes=int(data.CommitTotal*data.PageSize),
            system_commit_limit_bytes=int(data.CommitLimit*data.PageSize),
            system_peak_commit_bytes=int(data.CommitPeak*data.PageSize),
            physical_available_bytes=int(data.PhysicalAvailable*data.PageSize))
    except Exception as exc:result['errors'].append('system_commit: '+repr(exc))
    try:
        # Query NULL obtains the immediate Job of this process, including workers.
        # Public classes 9/13: ExtendedLimit / LimitViolationInformation.
        class Basic(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),
                ('LimitFlags',w.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),
                ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',w.DWORD),
                ('Affinity',ctypes.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_uint64) for n in ('ReadOperationCount','WriteOperationCount',
                'OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO)]+[
                (n,ctypes.c_size_t) for n in ('ProcessMemoryLimit','JobMemoryLimit',
                                             'PeakProcessMemoryUsed','PeakJobMemoryUsed')]
        class Violation(ctypes.Structure):
            _fields_=[('LimitFlags',w.DWORD),('ViolationLimitFlags',w.DWORD)]+[
                (n,ctypes.c_uint64) for n in ('IoReadBytes','IoReadBytesLimit','IoWriteBytes',
                'IoWriteBytesLimit','PerJobUserTime','PerJobUserTimeLimit','JobMemory','JobMemoryLimit')]+[
                ('RateControlTolerance',ctypes.c_int),('RateControlToleranceLimit',ctypes.c_int)]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=w.HANDLE
        kernel.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)]
        kernel.IsProcessInJob.restype=w.BOOL
        present=w.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(present)):
            raise ctypes.WinError(ctypes.get_last_error())
        result['in_windows_job']=bool(present.value)
        if present.value:
            query=kernel.QueryInformationJobObject
            query.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
            query.restype=w.BOOL
            extended=Extended()
            if not query(None,9,ctypes.byref(extended),ctypes.sizeof(extended),None):
                raise ctypes.WinError(ctypes.get_last_error())
            result.update(job_peak_commit_bytes=int(extended.PeakJobMemoryUsed),
                job_commit_limit_bytes=int(extended.JobMemoryLimit)
                if extended.BasicLimitInformation.LimitFlags&0x200 else None)
            violation=Violation()
            if not query(None,13,ctypes.byref(violation),ctypes.sizeof(violation),None):
                raise ctypes.WinError(ctypes.get_last_error())
            result['job_commit_bytes']=int(violation.JobMemory)
    except Exception as exc:result['errors'].append('job_commit: '+repr(exc))
    return result


@contextmanager
def record_stage(record, stage, **fields):
    """Record the failing substage without catching or replacing its exception."""
    started=time.monotonic()
    record(stage+'_start',resources=resource_snapshot(),**fields)
    try:yield
    except BaseException as exc:
        record(stage+'_error',seconds=time.monotonic()-started,resources=resource_snapshot(),
               error=repr(exc),error_type=type(exc).__name__,**fields)
        raise
    else:record(stage+'_complete',seconds=time.monotonic()-started,
                resources=resource_snapshot(),**fields)


class _Journal:
    def __init__(self,path):
        self.path=assert_new_file(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic();self.events=[];self.failed_stage=None

    def record(self,stage,**fields):
        row=dict(stage=stage,elapsed_seconds=time.monotonic()-self.started,**fields)
        self.events.append(row)
        if stage.endswith('_error'):self.failed_stage=stage[:-6]
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(normalize(row),ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        from supervised import publish_progress
        publish_progress('iqae_diagnostic_'+stage,scope='iqae_diagnostic',**fields)


def _hash_file(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def export_qpy(circuit,path):
    """Write one gzip QPY, streaming the container and hashing its exact bytes."""
    from qiskit import qpy
    path=assert_new_file(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed:
            qpy.dump(circuit,compressed)
    return {'path':path.name,'sha256':_hash_file(path),'bytes':path.stat().st_size,
            'format':'qpy+gzip','qpy_version':qpy.QPY_VERSION,'reader':QPY_READER}


def load_qpy(path,expected_sha256=None):
    """Load QPY only; returned worker pickle files are never read by this module."""
    from qiskit.qpy import common,formats,type_keys,binary_io
    path=Path(path)
    if path.suffixes[-2:]!=['.qpy','.gz']:raise ValueError('Expected a .qpy.gz circuit export')
    if expected_sha256 is not None and _hash_file(path)!=expected_sha256:
        raise ValueError('QPY archive byte checksum mismatch')
    # The locked Qiskit Rust QPY reader overflows on MCX ctrl_state > 32 bits.
    # Its shipped Python reader preserves these exact circuits. Parse the small
    # documented header explicitly, then use that reader, including nested defs.
    # No process-global monkeypatch, dependency update, or circuit rewrite.
    with gzip.open(path,'rb') as stream:
        header=formats.FILE_HEADER_V10._make(struct.unpack(formats.FILE_HEADER_V10_PACK,
            stream.read(formats.FILE_HEADER_V10_SIZE)))
        if header.preface!=b'QISKIT' or not 13<=header.qpy_version<=min(17,common.QPY_VERSION):
            raise ValueError('Expected supported QPY version 13 through 17')
        if header.num_programs!=1:raise ValueError('Expected one circuit per QPY archive')
        if common.read_type_key(stream)!=type_keys.Program.CIRCUIT:
            raise ValueError('QPY payload must contain a circuit')
        if header.qpy_version>=16:
            offset=struct.unpack(formats.CIRCUIT_TABLE_ENTRY_PACK,
                                 stream.read(formats.CIRCUIT_TABLE_ENTRY_SIZE))[0]
            if offset<stream.tell():raise ValueError('Invalid QPY circuit offset')
            stream.seek(offset)
        return binary_io.read_circuit(stream,header.qpy_version,
            use_symengine=header.symbolic_encoding==type_keys.SymExprEncoding.SYMENGINE,
            use_rust=False)


def _exact(value):
    value=normalize(value)
    if isinstance(value,float):return {'value':value,'hex':value.hex()}
    if isinstance(value,list):return [_exact(v) for v in value]
    if isinstance(value,dict):return {k:_exact(v) for k,v in value.items()}
    return value


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def _layout(circuit):
    layout=circuit.layout
    if layout is None:return None
    return {'initial_index_layout':layout.initial_index_layout(filter_ancillas=False),
        'final_index_layout':layout.final_index_layout(filter_ancillas=False),
        'routing_permutation':layout.routing_permutation(),
        'input_qubit_mapping':sorted([(str(bit),index) for bit,index in layout.input_qubit_mapping.items()],
                                      key=lambda row:row[1])}


def compare_circuits(left,right,max_differences=12):
    """Stream exact comparison; retain only bounded examples, never gate JSON dumps.

    Counts are top-level positions per category. A definitions difference refers
    to the full recursive canonical definition fingerprint at that position.
    Numeric examples include IEEE float.hex, including signed zero.
    """
    if type(max_differences) is not int or max_differences<1:raise ValueError('max_differences must be positive')
    keys=('operations','qubits','clbits','params','conditions','definitions',
          'global_phase','registers','layout','instruction_count')
    counts={k:0 for k in keys};first=[];memo=OrderedDict()
    def note(field,a,b,index=None):
        if _encoded(a)==_encoded(b):return
        counts[field]+=1
        if len(first)<max_differences:
            row={'field':field,'instruction_index':index,'left':a,'right':b}
            first.append(row)
            return row
    note('global_phase',_exact(float(left.global_phase)),_exact(float(right.global_phase)))
    note('registers',{'q':[(r.name,len(r)) for r in left.qregs],
                     'c':[(r.name,len(r)) for r in left.cregs],'nq':left.num_qubits,'nc':left.num_clbits},
                    {'q':[(r.name,len(r)) for r in right.qregs],
                     'c':[(r.name,len(r)) for r in right.cregs],'nq':right.num_qubits,'nc':right.num_clbits})
    note('layout',_layout(left),_layout(right))
    note('instruction_count',len(left.data),len(right.data))
    def definition_hash(op):
        definition=op.definition
        if definition is None:return None
        key=id(definition)
        if key not in memo:
            memo[key]=(definition,circuit_fingerprint(definition))
            if len(memo)>256:memo.popitem(last=False)
        else:memo.move_to_end(key)
        return memo[key][1]
    def fields(qc,ins):
        op=ins.operation
        return {'operations':op.name,'qubits':[qc.find_bit(q).index for q in ins.qubits],
                'clbits':[qc.find_bit(c).index for c in ins.clbits],
                'params':_exact(list(op.params)),
                'conditions':str(getattr(op,'condition',None)),
                'definitions':definition_hash(op)}
    differing_positions=0
    for index in range(max(len(left.data),len(right.data))):
        a=fields(left,left.data[index]) if index<len(left.data) else {}
        b=fields(right,right.data[index]) if index<len(right.data) else {}
        if a!=b:differing_positions+=1
        for key in keys[:6]:
            example=note(key,a.get(key),b.get(key),index)
            if example is not None and key=='definitions' and a.get(key) and b.get(key):
                nested=compare_circuits(left.data[index].operation.definition,
                                        right.data[index].operation.definition,max_differences=1)
                example['nested_first_difference']=nested['first_differences'][0] if nested['first_differences'] else None
    return {'equal':not any(counts.values()),'difference_counts':counts,
        'differing_instruction_positions':differing_positions,'first_differences':first,
        'counts_scope':'top-level instruction positions; definitions compare recursive canonical identities',
        'first_differences_limit':max_differences,
        'operation_counts':{'left':dict(left.count_ops()),'right':dict(right.count_ops())}}


def compare_qpy(left,right,max_differences=12):
    a=load_qpy(left);b=load_qpy(right)
    result=compare_circuits(a,b,max_differences)
    result.update(left_sha256=_hash_file(left),right_sha256=_hash_file(right),
        left_fingerprint=circuit_fingerprint(a),right_fingerprint=circuit_fingerprint(b),
        same_archive_bytes=_hash_file(left)==_hash_file(right))
    return result


def historical_identity_comparison(report):
    """Compare to audited cloud hashes; mismatches carry no equivalence claim."""
    meta=normalize(report.get('case_metadata') or {})
    name=Path(str(meta.get('model_locator',''))).stem
    profile=meta.get('profile',{})
    profile=profile.get('name') if isinstance(profile,dict) else profile
    expected={
        ('q_uniform_small',2026110803):{
            'marked_A':'478c92f685bdc1e247384c19544c24465287bc077d4be7a550870cc58351e6ab',
            'Q':'cc1413a3bf91a8176e86aec5fa536a23cf044e9a5a48691c38719da148e54dc4',
            'k0_compiled_measured':'dcffd145fa9313ace120ab7888fccfecc384d69ef00f86aa0178302b0306a2ab',
            'k6_compiled_measured':'c26d8c9795c4774d89c4e491afcaa21480b126ce14065847a5471a70994fc362'},
        ('q_tau19_two_cell',2026110804):{
            'k0_compiled_measured':'7d975f7ece597af9961d7edcaecd3477d7d6ca342d69bfb9952bf79ded3b7e8f',
            'k5_compiled_measured':'b288498fceaf0a382987ebdcb4f4e8ce8ed3f09c67ef73f84b47f6370c213e53'}}
    hashes=expected.get((name,report.get('seed')))
    base={'numerical_equivalence_certified':False,
        'source':'Audited 2026-11-09 cloud returns uniform_T3_b128_01 / tau19_cells2_T3_b128_01',
        'scope':'Exact canonical-json-v1 identity only. No returned QPY was present in those runs; a hash mismatch alone cannot locate its cause or certify physics.'}
    if hashes is None or meta.get('steps')!=3 or profile!='baseline':
        return dict(base,status='not_applicable',rows=[])
    actual={k:v['fingerprint'] for k,v in report['components'].items()}
    actual.update({f"k{row['power']}_compiled_measured":row['circuits']['compiled_measured']['fingerprint']
                   for row in report['powers']})
    rows=[{'component':key,'historical_cloud_fingerprint':value,'current_fingerprint':actual[key],
           'matches':value==actual[key]} for key,value in hashes.items() if key in actual]
    return dict(base,status='all_available_match' if rows and all(row['matches'] for row in rows)
                else 'mismatch_unresolved' if rows else 'not_evaluated',rows=rows)


def sparse_probability(circuit,*,objective_index,max_sparse_states=60000):
    from reference_engine import evolve
    if type(max_sparse_states) is not int or max_sparse_states<1:
        raise ValueError('max_sparse_states must be positive')
    result={'engine':'independent actual-gate sparse complex amplitudes','pruning':False,
        'renormalization':False,'wall_deadline_seconds':None,'max_sparse_states':max_sparse_states,
        'objective_probability':None}
    try:state,runtime=evolve(circuit,timeout=None,max_states=max_sparse_states)
    except MemoryError as exc:
        result.update(status='state_budget_exceeded',error=str(exc));return result
    work=[circuit.find_bit(q).index for r in circuit.qregs if r.name=='work' for q in r]
    mask=sum(1<<b for b in work)
    result.update(status='complete',runtime=runtime,
        objective_probability=[math.fsum(abs(a)**2 for i,a in state.items() if (i>>objective_index)&1==v) for v in (0,1)],
        norm=math.fsum(abs(a)**2 for a in state.values()),
        work_nonzero=math.fsum(abs(a)**2 for i,a in state.items() if i&mask))
    return result


def _stats(circuit):
    return {'num_qubits':circuit.num_qubits,'num_clbits':circuit.num_clbits,
        'depth':circuit.depth(),'gates':len(circuit.data),'operation_counts':dict(circuit.count_ops()),
        'global_phase':_exact(float(circuit.global_phase)),'layout':_layout(circuit)}


def _export_component(circuit,name,out,journal,**fields):
    with record_stage(journal.record,'fingerprint',component=name,**fields):fingerprint=circuit_fingerprint(circuit)
    with record_stage(journal.record,'circuit_stats',component=name,**fields):stats=_stats(circuit)
    with record_stage(journal.record,'serialization',component=name,**fields):artifact=export_qpy(circuit,out/(name+'.qpy.gz'))
    return {**artifact,'fingerprint':fingerprint,'fingerprint_format':'canonical-json-v1',**stats}


def _objective_probability(compiled,sampler,journal,power):
    import numpy as np
    from qiskit_aer.library import save_matrix_product_state,save_probabilities,save_density_matrix
    from mps_numerical import mps_norm
    diagnostic=compiled.copy()
    work=[diagnostic.find_bit(q).index for r in diagnostic.qregs if r.name=='work' for q in r]
    diagnostic.save_matrix_product_state(label='iqae_mps_before')
    diagnostic.save_probabilities([sampler._obj],label='iqae_objective')
    diagnostic.save_density_matrix([sampler._obj],label='iqae_objective_density',unnormalized=True)
    for bit in work:diagnostic.save_probabilities([bit],label='iqae_work_'+str(bit))
    with record_stage(journal.record,'backend_submit',power=power):
        job=sampler._backend.run(diagnostic,shots=1,seed_simulator=sampler._seed)
    with record_stage(journal.record,'backend_result',power=power):
        result=job.result()
        if not result.success:raise RuntimeError(str(result.status))
        data=result.data(0)
    with record_stage(journal.record,'probability_readout',power=power):
        p=[float(v) for v in data['iqae_objective']]
        density=np.asarray(data['iqae_objective_density'])
        norm=mps_norm(data['iqae_mps_before'])
        perwork={str(b):float(data['iqae_work_'+str(b)][1]) for b in work}
        metadata=result.results[0].metadata
        keep=('method','device','num_qubits','active_input_qubits','input_qubit_map',
              'remapped_qubits','matrix_product_state_max_bond_dimension',
              'matrix_product_state_truncation_threshold','mps_lapack',
              'parallel_state_update','parallel_shots')
        return {'objective_probability':p,'probability_norm':math.fsum(p),
            'tensor_norm_before':norm,'density_trace':normalize(complex(np.trace(density))),
            'density_diagonal':normalize(list(np.diag(density))),
            'density_vs_saved_max_abs':max(float(abs(density[i,i]-p[i])) for i in (0,1)),
            'tensor_vs_saved_norm_abs':abs(norm-math.fsum(p)),
            'work_qubit_p1':perwork,'work_nonzero_union_upper_sum':math.fsum(perwork.values()),
            'work_measure':'per-work-bit marginal; sum is a union upper bound for a normalized state, not joint P(work!=0)',
            'backend_metadata':normalize({k:metadata[k] for k in keep if k in metadata}),
            'scope':'Instrumented actual compiled premeasurement circuit. MPS snapshots may change Aer active-qubit mapping. Raw saves and independent floating tensor norm; no clipping, pruning, normalization or IQAE estimation.'}


def run_diagnostic(*,A_circuit,status_qubits,g_qubits,spec,powers,outdir,seed,
                   mode='compile',backend_options=None,max_sparse_states=60000,
                   compare_to=None,case_metadata=None,max_compiled_gates=4000000):
    """Compile/export a finite explicit power schedule through production methods.

    The caller owns process supervision and the memory limit. Powers are processed
    serially; measured compilation is the sole transpilation per power. Removing
    its final measurement yields the exported actual premeasurement compiled form.
    """
    import iqae
    from ae_runtime import BudgetExceeded
    if mode not in ('compile','probability','sparse'):raise ValueError('Unknown IQAE diagnostic mode')
    powers=list(powers)
    if not powers or any(type(k) is not int or k<0 for k in powers) or len(set(powers))!=len(powers):
        raise ValueError('A nonempty unique schedule of nonnegative integer powers is required')
    if type(max_compiled_gates) is not int or max_compiled_gates<1:raise ValueError('max_compiled_gates must be positive')
    if type(max_sparse_states) is not int or max_sparse_states<1:raise ValueError('max_sparse_states must be positive')
    if type(seed) is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    out=validate_output_path(outdir);out.mkdir(parents=True,exist_ok=True)
    if (out/'diagnostic.json').exists() or (out/'stages.jsonl').exists():raise FileExistsError('Use a new diagnostic run directory')
    comparison_root=Path(compare_to).resolve() if compare_to else None
    if comparison_root is not None and not comparison_root.is_dir():raise ValueError('compare_to must be an existing export directory')
    journal=_Journal(out/'stages.jsonl')
    report=dict(schema_version=1,phase='iqae_diagnostic',mode=mode,status='running',
        estimation_performed=False,estimate=None,confidence_interval=None,precision_target_met=False,
        backend_submitted=False,seed=seed,requested_powers=powers,powers=[],components={},
        case_metadata=case_metadata,compile_options={'basis_gates':['rz','sx','x','cx'],
        'optimization_level':0,'seed_transpiler':seed},software_versions=software_versions(),
        sources=source_manifest(),max_compiled_gates=max_compiled_gates,max_sparse_states=max_sparse_states,
        scope='Bounded circuit identity and numerical diagnostic only; no IQAE sampling estimate or statistical stopping.')
    try:
        with record_stage(journal.record,'grover_build'):
            Q,obj,marker=iqae.build_grover_operator_for_spec(A_circuit=A_circuit,
                status_qubits=status_qubits,g_qubits=g_qubits,spec=spec)
        sampler=iqae._PubsToAerObjectiveSampler(shots=1,objective_qubit_index=obj[0],
            seed=seed,verbose=False,backend_options=backend_options)
        report.update(objective_index=obj[0],marker_meta=marker,reflection_qubits=list(Q.reflection_qubits),
            backend_settings=sampler.effective_backend_options,
            measurement_mapping={'objective_index':obj[0],'classical_register':'cobj','width':1})
        # Match production _start's fingerprint materialization order first.
        for name,circuit in (('marked_A',Q.state_preparation),('Q',Q),('A',A_circuit)):
            report['components'][name]=_export_component(circuit,name,out,journal)
        for power in powers:
            with record_stage(journal.record,'power_build',power=power):
                raw=iqae._build_power_circuit(Q,power)
                raw.metadata={**(raw.metadata or {}),'grover_power':power}
            with record_stage(journal.record,'measurement_wrap',power=power):
                measured=sampler._force_measure_objective_only(raw)
            with record_stage(journal.record,'transpile',power=power):
                compiled=sampler._compile_for_aer(measured,call_id=1,pub_id=power)
            with record_stage(journal.record,'compiled_gate_budget',power=power):
                if len(compiled.data)>max_compiled_gates:
                    raise BudgetExceeded(f'compiled circuit has {len(compiled.data)} gates > {max_compiled_gates}')
            del measured
            row={'power':power,'circuits':{}}
            for name,circuit in (('power_raw',raw),('compiled_measured',compiled)):
                row['circuits'][name]=_export_component(circuit,f'k{power}_{name}',out,journal,power=power)
            del circuit
            with record_stage(journal.record,'measurement_remove',power=power):
                if compiled.data[-1].operation.name!='measure' or compiled.count_ops().get('measure')!=1:
                    raise ValueError('Expected the single production objective measurement at the circuit end')
                premeasurement=compiled.copy()
                premeasurement.data.pop()
            del compiled
            row['premeasurement_derivation']='Exact compiled copy with its single terminal objective measure removed; unused cobj retained. No second transpile or DAG pass.'
            row['circuits']['compiled_premeasurement']=_export_component(
                premeasurement,f'k{power}_compiled_premeasurement',out,journal,power=power)
            if mode=='sparse':
                with record_stage(journal.record,'sparse_evolution',power=power):
                    row['sparse']=sparse_probability(raw,objective_index=obj[0],max_sparse_states=max_sparse_states)
                with record_stage(journal.record,'compiled_sparse_evolution',power=power):
                    row['compiled_sparse']=sparse_probability(premeasurement,objective_index=obj[0],max_sparse_states=max_sparse_states)
            elif mode=='probability':
                report['backend_submitted']=True
                row['probability']=_objective_probability(premeasurement,sampler,journal,power)
            report['powers'].append(row)
            del raw,premeasurement
            with record_stage(journal.record,'power_cleanup',power=power):gc.collect()
        if comparison_root is not None:
            report['comparisons']={}
            artifacts=list(report['components'].values())+[
                artifact for row in report['powers'] for artifact in row['circuits'].values()]
            for artifact in artifacts:
                filename=artifact['path'];other=comparison_root/filename
                if not other.is_file():
                    report['comparisons'][filename]={'status':'missing_comparison_circuit'};continue
                with record_stage(journal.record,'circuit_compare',component=filename):
                    report['comparisons'][filename]=compare_qpy(out/filename,other)
        report['status']='complete'
        if any(row.get(key,{}).get('status')=='state_budget_exceeded'
               for row in report['powers'] for key in ('sparse','compiled_sparse')):
            report['status']='state_budget_exceeded'
    except Exception as exc:
        report.update(status='memory_limit' if isinstance(exc,MemoryError) else getattr(exc,'status','failed'),
            error=repr(exc),error_type=type(exc).__name__,failed_stage=journal.failed_stage)
    report['elapsed_seconds']=time.monotonic()-journal.started
    report['historical_identity']=historical_identity_comparison(report)
    report['stages_file']='stages.jsonl'
    report['stage_summary']=[{k:v for k,v in e.items() if k!='resources'} for e in journal.events
                             if e['stage'].endswith(('_complete','_error'))]
    write_json_new(out/'diagnostic.json',report)
    return report
