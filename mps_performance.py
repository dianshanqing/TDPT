"""One finite compiled-circuit prefix, with measured resource and identity evidence.

This is a performance diagnostic. Prefix observables are not the final transport
response, and agreement with another MPS run is not an independent accuracy proof.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import threading
import time
import traceback
import sys

from models import load_document, load_model, resolve_path
from runtime_identity import (apply_backend_options, check_identity, circuit_fingerprint,
                              digest, normalize, validate_output_path, write_json_new)
from supervised import publish_progress
from validation import (_validated_cases, _probe_settings, _probe_backend_options,
                        _probe_summary_row, _identity, _positive_integer, _probe_seed,
                        TOL, ORDER)


CPU_LIMITATIONS = ('Sampled process-tree CPU is a lower bound: exited children between '
                  'observations may be missed. The outer supervisor Windows Job lifetime '
                  'CPU record is authoritative. No SVD or tensor-contraction profiling.')
_PREFIX_CALLS=0


def _host_identity():
    """Comparable timing requires the same host and CPU affinity, not versions alone."""
    import psutil
    machine_key=None
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\Cryptography') as key:
            machine_key=winreg.QueryValueEx(key,'MachineGuid')[0]
    return {'host_name':platform.node(),'machine_key_sha256':digest(machine_key) if machine_key else None,
            'processor':platform.processor(),'machine':platform.machine(),
            'cpu_affinity':sorted(psutil.Process().cpu_affinity()),
            'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False)}


class _Resources:
    """Low-frequency RSS sampling; retain last CPU observation of exited children."""
    def __init__(self):
        import psutil
        self.psutil=psutil; self.process=psutil.Process(); self.lock=threading.Lock()
        self.stop=threading.Event(); self.cpu={}; self.main_cpu=None
        self.peak_rss=0; self.peak_tree_rss=0; self.samples=0; self.errors=0
        self.thread=threading.Thread(target=self._poll,daemon=True)

    def sample(self):
        with self.lock:
            try:
                processes=[self.process]+self.process.children(recursive=True)
            except (self.psutil.Error,OSError):
                self.errors+=1; processes=[self.process]
            tree_rss=0
            for proc in processes:
                try:
                    key=(proc.pid,proc.create_time()); cpu=proc.cpu_times(); memory=proc.memory_info()
                    seconds=cpu.user+cpu.system
                    self.cpu[key]=max(self.cpu.get(key,0.),seconds); tree_rss+=memory.rss
                    if proc.pid==self.process.pid:
                        self.main_cpu=seconds
                        self.peak_rss=max(self.peak_rss,getattr(memory,'peak_wset',memory.rss))
                except (self.psutil.Error,OSError):self.errors+=1
            self.peak_tree_rss=max(self.peak_tree_rss,tree_rss); self.samples+=1
            return {'process_cpu':self.main_cpu,'tree_cpu':sum(self.cpu.values()) if self.cpu else None}

    def _poll(self):
        while not self.stop.wait(.1):self.sample()

    def __enter__(self):
        self.sample(); self.thread.start(); return self

    def __exit__(self,*args):
        self.stop.set(); self.thread.join(); self.sample()

    def record(self):
        return {'process_peak_rss_bytes':self.peak_rss or None,
                'sampled_tree_peak_rss_bytes':self.peak_tree_rss or None,
                'sampling_interval_seconds':.1,'samples':self.samples,
                'observation_errors':self.errors,'tree_peak_is_sampled_lower_bound':True}


class _Stages:
    def __init__(self,resources):self.resources=resources;self.rows={}

    @contextmanager
    def measure(self,name):
        before=self.resources.sample(); start=time.perf_counter()
        publish_progress('prefix_'+name,event='start')
        print(json.dumps({'stage':name,'event':'start','wall_unix':time.time()}),flush=True)
        outcome='complete'
        try:yield
        except BaseException:
            outcome='failed';raise
        finally:
            elapsed=time.perf_counter()-start;after=self.resources.sample()
            def delta(key):
                return (max(0.,after[key]-before[key])
                        if after[key] is not None and before[key] is not None else None)
            self.rows[name]={'wall_seconds':elapsed,'process_cpu_seconds':delta('process_cpu'),
                'tree_cpu_seconds':delta('tree_cpu'),'cpu_coverage':'sampled_lower_bound',
                'status':outcome}
            publish_progress('prefix_'+name,event='end',**self.rows[name])
            print(json.dumps({'stage':name,'event':'end',**self.rows[name]}),flush=True)


def _bond_statistics(log,cap,*,isolated_worker=False):
    maximum=None; count=0; at_cap=0
    if isinstance(log,str):
        for match in re.finditer(r'BD=\[([^\]]*)\]',log):
            values=[int(value) for value in match.group(1).split()]
            if not values:continue
            largest=max(values);count+=1
            maximum=largest if maximum is None else max(maximum,largest)
            at_cap+=int(largest>=cap)
    return {'max_observed_bond':maximum if isolated_worker else None,
            'max_log_bond':maximum,'BD_records':count,'records_at_cap':at_cap,
            'bond_cap':cap,'near_cap_threshold':math.ceil(.75*cap),
            'representative_bond_near_cap':isolated_worker and maximum is not None and maximum>=math.ceil(.75*cap),
            'reached_cap':isolated_worker and maximum is not None and maximum>=cap,
            'current_run_attribution_verified':isolated_worker,
            'log_scope':'fresh supervised prefix worker' if isolated_worker else 'process-cumulative; current-run attribution unverified',
            'interpretation':'Aer MPS_log_data accumulates across calls in one process. Fresh main.py prefix workers make its range attributable; direct in-process diagnostics do not. Record counts are not time fractions; no function-level profiling.'}


def _library_identity(records):
    fields=('path','owner','internal_api','config','version','parallel','threading_layer')
    return sorted(({key:record.get(key) for key in fields} for record in records),
                  key=lambda record:record.get('path') or '')


def _signature(row):
    check_identity(row['identity'])
    if row.get('record_type') not in ('prefix_result','performance_summary'):
        raise ValueError('not a prefix-performance record')
    keys=('original_A_fingerprint','compiled_A_fingerprint','prefix_fingerprint',
          'diagnostic_fingerprint','prefix_gates','full_compiled_gates','config',
          'profile','steps','seed','bond','truncation_threshold','register_map','status_order')
    result={key:row[key] for key in keys}
    result['host_identity']=row['host_identity']
    result['identity_fields']={key:row['identity'][key]
        for key in ('config','actual_profile','compile','measurement','sources','versions','phase','extra')}
    ignored={'max_parallel_threads','mps_omp_threads','mps_lapack'}
    result['backend_numeric_options']={key:value for key,value in row['backend']['effective'].items()
                                       if key not in ignored}
    result['libraries']=_library_identity(row['numerical_libraries_configured'])
    return normalize(result)


def _file_sha256(path):
    from result_storage import content_digest
    return content_digest(path)['sha256']


def _load_reference(path):
    from result_storage import read_json
    path=resolve_path(path)
    row=read_json(path, encoding='utf-8-sig')
    if row.get('record_type')=='performance_summary':
        original=path.parent/row['result_path']
        if _file_sha256(original)!=row['result_file_sha256']:
            raise ValueError('reference mismatch: authoritative prefix result hash')
        row=read_json(original, encoding='utf-8-sig')
        path=original
    if row.get('execution_status')!='passed':
        raise ValueError('reference mismatch: baseline execution did not pass')
    # The raw log is not required to compare identical observables.
    row.get('metadata',{}).pop('MPS_log_data',None)
    return row,{'path':str(path),'sha256':_file_sha256(path)}


def _observables(row):
    status=row['probabilities'];work=row['work_bit_probabilities']
    if not isinstance(status,list) or len(status)!=4 or not isinstance(work,list):
        raise ValueError('Expected four status probabilities and work marginals')
    if len(work)!=len(row['register_map']['work']):raise ValueError('Work observable layout mismatch')
    values=status+work
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
           or value < -TOL or value > 1+TOL for value in values):
        raise ValueError('Backend returned invalid probability observables')
    return values,abs(math.fsum(status)-1.)


def _compare(row,reference):
    actual,norm=_observables(row);expected,ref_norm=_observables(reference)
    if len(actual)!=len(expected):raise ValueError('reference mismatch: observable layout')
    error=max(abs(a-b) for a,b in zip(actual,expected))
    d_reference=reference['probabilities'][1]
    readouts_ok=all(not item.get('readout_diagnostics_enabled') or
        (item.get('readout_diagnostics',{}).get('consistency_status')=='passed' and
         item.get('readout_diagnostics',{}).get('normalization_status')=='passed')
        for item in (row,reference))
    def controls(item):
        opts=item['backend']['effective']
        return {'aer_threads':opts['max_parallel_threads'],'mps_threads':opts['mps_omp_threads'],
                'blas_threads':item['cpu']['blas_threads'],'mps_lapack':opts['mps_lapack']}
    left=controls(reference);right=controls(row)
    changed={k:{'reference':left[k],'current':right[k]} for k in left if left[k]!=right[k]}
    mapping_keys=('active_input_qubits','input_qubit_map','num_qubits')
    same_mapping=all(k in row['metadata'] and k in reference['metadata'] and
        row['metadata'][k]==reference['metadata'][k] for k in mapping_keys)
    controlled=len(changed)<=1 and same_mapping
    numerical_passed=error<TOL and norm<TOL and ref_norm<TOL and readouts_ok
    passed=numerical_passed and controlled
    reference_seconds=reference['evolution_and_observable_read_seconds']
    seconds=row['evolution_and_observable_read_seconds']
    observed_ratio=(reference_seconds/seconds if all(isinstance(t,(int,float)) and
        math.isfinite(t) and t>0 for t in (reference_seconds,seconds)) else None)
    return {'status':'passed' if passed else 'failed','comparable':True,
        'absolute_tolerance':TOL,'max_abs_observable_error':error,
        'D_relative_error':None,
        'status_index1_relative_error':abs(row['probabilities'][1]-d_reference)/abs(d_reference) if d_reference else None,
        'current_norm_residual':norm,'reference_norm_residual':ref_norm,
        'additional_readouts_passed':readouts_ok,
        'numerical_comparison_status':'passed' if numerical_passed else 'failed',
        'changed_controls':changed,'one_factor_controlled':len(changed)<=1,
        'same_active_qubit_mapping':same_mapping,
        'baseline_status':'reference_only','baseline_scientific_accuracy_certified':False,
        'observed_evolution_speed_ratio':observed_ratio,
        'speed_ratio_scope':'Measured reference/current wall time; independent of accuracy and single-factor acceptance.',
        'speedup_accepted':passed and observed_ratio is not None,
        'evolution_speedup':observed_ratio if passed else None,
        'interpretation':'MPS prefix repeat agreement only; final transport accuracy is not assessed.'}


def run_prefix_probe(spec,outdir,cpu_budget=1,*,aer_threads=1,mps_threads=1,
                     blas_threads=1,prefix_gates,reference_path=None):
    """Build full A but evolve exactly one strict prefix, one bond and one threshold."""
    global _PREFIX_CALLS
    _PREFIX_CALLS+=1
    isolated_worker=(os.environ.get('VRQO_SUPERVISED')=='1' and _PREFIX_CALLS==1 and
                     Path(sys.argv[0]).name=='main.py' and '--phase' in sys.argv and
                     sys.argv[sys.argv.index('--phase')+1]=='mps_probe')
    spec=load_document(spec);cases=_validated_cases(spec);bonds,thresholds,_=_probe_settings(spec)
    if len(cases)!=1 or len(bonds)!=1 or len(thresholds)!=1:
        raise ValueError('Prefix performance requires exactly one case, bond and threshold')
    if spec.get('counts_check') is not False:
        raise ValueError('Prefix performance requires explicit counts_check=false')
    readout_diagnostics=spec.get('readout_diagnostics',False)
    if not isinstance(readout_diagnostics,bool):raise ValueError('readout_diagnostics must be boolean')
    prefix_gates=_positive_integer(prefix_gates,'prefix_gates')
    blas_threads=_positive_integer(blas_threads,'blas_threads')
    if os.environ.get('VRQO_DEADLINE_MONOTONIC'):
        raise ValueError('Inherited deadline is incompatible with an unlimited finite-prefix probe')
    case=cases[0];cfg=load_model(case['config']);profile=case.get('profile','compressed')
    seed=_probe_seed(spec.get('seed',20261030));bond=bonds[0];threshold=thresholds[0]
    extras=dict(spec.get('backend_options',{}));extras.update(case.get('backend_options',{}))
    if extras.get('mps_log_data',True) is not True:
        raise ValueError('Prefix performance requires mps_log_data=true to observe actual bonds')
    options=_probe_backend_options(cpu_budget,aer_threads,mps_threads,extras,bond,threshold,seed)
    if options['max_parallel_threads']*options['mps_omp_threads']*blas_threads>cpu_budget:
        raise ValueError('aer_threads*mps_threads*blas_threads exceeds the total CPU budget')
    output=validate_output_path(resolve_path(outdir))
    from runtime_identity import assert_new_file
    for filename in ('prefix_result.json','performance_summary.json'):
        assert_new_file(output/filename)
    output.mkdir(parents=True,exist_ok=True)
    row={'schema_version':1,'record_type':'prefix_result','status':'failed','execution_status':'failed','passed':False,
        'science_status':'not_assessed_prefix','normalization_status':'not_evaluated',
        'formal_D_accuracy_status':'not_assessed_prefix',
        'formal_D_relative_error':None,
        'work_accuracy_status':'not_applicable_intermediate_compiled_prefix',
        'status_probability_semantics':'Intermediate status bits, not final D/A/B/S; do not compare to full T15 reference.',
        'host_identity':_host_identity(),'readout_diagnostics_enabled':readout_diagnostics,
        'comparison':{'status':'not_evaluated','comparable':False,'evolution_speedup':None,
                      'observed_evolution_speed_ratio':None,'speedup_accepted':False,
                      'baseline_scientific_accuracy_certified':False},
        'case':case,'config':asdict(cfg),'profile':profile,'steps':case['steps'],
        'seed':seed,'bond':bond,'truncation_threshold':threshold,'prefix_gates':prefix_gates,
        'status_order':ORDER,'counts_check':False,'probabilities':None,'work_bit_probabilities':None,
        'evolution_and_observable_read_seconds':None,'cpu_measurement_limitations':CPU_LIMITATIONS,
        'cpu':{'requested_total_budget':cpu_budget,'workers':1,'aer_threads':aer_threads,
               'mps_threads':mps_threads,'blas_threads':blas_threads},
        'write_timing_scope':'Raw prefix_result serialization and summary reference hashing; final summary serialization is logged separately.',
        'separate_evolution_and_read_timing':'Not exposed by this Aer API; the combined call is timed.',
        'independent_prefix_accuracy':'not_assessed; no full-final DP or expensive sparse reference is used'}
    with _Resources() as resources:
        stages=_Stages(resources);row['stages']=stages.rows
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
            from qiskit_aer.library import save_probabilities  # Registers circuit save methods.
            from reference_engine import registers
            from unitary_circuit import build_unitary_circuit_qiskit
            import numerical_runtime
            with stages.measure('build'):
                circuit,_=build_unitary_circuit_qiskit(cfg,case['steps'],kernel_profile=profile)
            with stages.measure('compile'):
                compiled=transpile(circuit,basis_gates=['u','cx'],optimization_level=0,
                                   seed_transpiler=917,num_processes=1)
                row['full_compiled_gates']=len(compiled.data)
                if prefix_gates>=len(compiled.data):
                    raise ValueError('prefix_gates must be strictly shorter than the complete compiled A')
                prefix=compiled.copy_empty_like();prefix.global_phase=compiled.global_phase
                for instruction in compiled.data[:prefix_gates]:prefix.append(instruction)
                regs=registers(prefix);instrumented=prefix.copy()
                if readout_diagnostics:
                    from mps_numerical import append_readout_after
                instrumented.save_probabilities(regs['status'],label='status_probability')
                for i,bit in enumerate(regs['work']):instrumented.save_probabilities([bit],label=f'work_{i}')
                if readout_diagnostics:append_readout_after(instrumented,regs['status'],include_mps=False)
                row.update(register_map=regs,prefix_global_phase=float(prefix.global_phase),
                    qubits=prefix.num_qubits,prefix_two_qubit_gates=sum(len(i.qubits)==2 for i in prefix.data))
            with stages.measure('fingerprints'):
                row.update(original_A_fingerprint=circuit_fingerprint(circuit),
                    compiled_A_fingerprint=circuit_fingerprint(compiled),
                    prefix_fingerprint=circuit_fingerprint(prefix),
                    diagnostic_fingerprint=circuit_fingerprint(instrumented))
            with stages.measure('backend_configuration'):
                sim=AerSimulator();row['backend']=apply_backend_options(sim,options)
                row['numerical_libraries_before']=numerical_runtime.numerical_libraries()
                row['blas_configuration']=numerical_runtime.configure_blas_threads(blas_threads)
                row['numerical_libraries_configured']=numerical_runtime.numerical_libraries()
                aer_libraries=[item for item in row['blas_configuration'] if item.get('owner')=='aer_private']
                if not aer_libraries or not all(item.get('applied') is True for item in aer_libraries):
                    raise RuntimeError('Aer BLAS effective thread configuration could not be verified')
                identity=_identity(cfg,case['steps'],profile,circuit,row['backend'],'mps_probe',
                                   row['original_A_fingerprint'])
                identity['phase']='mps_prefix_probe'
                identity['compile'].update(prefix_gates=prefix_gates,
                    full_compiled_gates=len(compiled.data),compiled_A_fingerprint=row['compiled_A_fingerprint'])
                identity['measurement'].update(observable_scope='compiled_prefix_status_and_work',
                    prefix_fingerprint=row['prefix_fingerprint'],diagnostic_fingerprint=row['diagnostic_fingerprint'])
                identity['identity_hash']=digest({k:v for k,v in identity.items() if k!='identity_hash'})
                row['identity']=identity
            baseline=None
            if reference_path is not None:
                with stages.measure('reference_validation'):
                    baseline,provenance=_load_reference(reference_path);row['reference_provenance']=provenance
                    try:matches=_signature(row)==_signature(baseline)
                    except (KeyError,TypeError,ValueError) as exc:
                        raise ValueError(f'reference mismatch: {exc}') from exc
                    if not matches:raise ValueError('reference mismatch: circuit/configuration/numerics/source/version/observables')
                    _observables(baseline)
            with stages.measure('evolution_and_reads'):
                tick=time.perf_counter()
                try:
                    result=sim.run(instrumented,shots=1).result()
                    row['evolution_and_observable_read_seconds']=time.perf_counter()-tick
                finally:
                    row['numerical_libraries_after']=numerical_runtime.numerical_libraries()
                if not result.success:raise RuntimeError(str(result.status))
            with stages.measure('result_extraction'):
                data=result.data();row['probabilities']=data['status_probability'].tolist()
                row['work_bit_probabilities']=[float(data[f'work_{i}'][1]) for i in range(len(regs['work']))]
                _,norm=_observables(row);row['normalization_residual']=norm
                row['normalization_status']='passed' if norm<TOL else 'failed'
                row['metadata']=normalize(result.results[0].metadata)
                if readout_diagnostics:
                    from mps_numerical import readout_record
                    row['readout_diagnostics']=readout_record(data,row['probabilities'],include_mps=False)
                row['bond_statistics']=_bond_statistics(row['metadata'].get('MPS_log_data'),bond,
                                                       isolated_worker=isolated_worker)
                row['comparison']=(_compare(row,baseline) if baseline is not None else {
                    'status':'reference_only','comparable':False,'absolute_tolerance':TOL,
                    'max_abs_observable_error':None,'D_relative_error':None,'evolution_speedup':None,
                    'observed_evolution_speed_ratio':None,'speedup_accepted':False,
                    'baseline_status':'reference_only','baseline_scientific_accuracy_certified':False})
                row.update(status='complete',execution_status='passed',passed=True)
        except Exception as exc:
            row.update(status='failed',execution_status='failed',error=str(exc),
                       error_type=type(exc).__name__,traceback=traceback.format_exc())
            if hasattr(exc,'records'):row['blas_configuration_failure']=exc.records
        row['memory']=resources.record()
        with stages.measure('write'):
            result_path=write_json_new(output/'prefix_result.json',row)
            report=_probe_summary_row(row,result_path)
            report.update(record_type='performance_summary',result_path=result_path.name,
                          result_file_sha256=_file_sha256(result_path))
        report['memory']=resources.record()
        print(json.dumps({'stage':'summary_write','event':'start'}),flush=True)
        tick=time.perf_counter();write_json_new(output/'performance_summary.json',report)
        print(json.dumps({'stage':'summary_write','event':'end','wall_seconds':time.perf_counter()-tick}),flush=True)
        publish_progress('prefix_complete',execution_status=report['execution_status'],
                         science_status=report['science_status'],comparison_status=report['comparison']['status'])
    return normalize(report)
