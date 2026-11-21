"""Compile only four components of the selected short oracle; never Q^k or MPS.

This follows tests/collect_precision_evidence.py. Compose counts only after the
single estimation attempt has ended. No resource inference from a macro-Q depth.
"""
from pathlib import Path
import sys,json,time
from complete_sparse_iqae import ROOT,DEFAULT_OUT,write_new
from runtime_identity import circuit_fingerprint,source_manifest,software_versions


def components():
    from qiskit import transpile
    from models import load_model
    from unitary_circuit import build_unitary_circuit_qiskit
    from iqae import OracleSpec,build_grover_operator_for_spec
    from reference_engine import registers
    from tests.collect_precision_evidence import resource
    from numerical_runtime import configure_blas_threads
    configure_blas_threads(1)
    saved=json.loads((DEFAULT_OUT/'circuit_identity.json').read_text(encoding='utf8'))
    a,_=build_unitary_circuit_qiskit(load_model('q_tau19_two_cell'),3,kernel_profile='baseline')
    regs=registers(a);q,_,_=build_grover_operator_for_spec(A_circuit=a,status_qubits=regs['status'],g_qubits=regs['g'],spec=OracleSpec(good='detect'))
    assert circuit_fingerprint(a)==saved['fingerprints']['A']
    assert circuit_fingerprint(q)==saved['fingerprints']['Q']
    options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=0,seed_transpiler=20261030,num_processes=1)
    rows={};start=time.perf_counter()
    for name,circuit in [('Atilde',q.state_preparation),('Atilde_inverse',q.state_preparation.inverse()),('S0',q.zero_reflection),('Sf',q.oracle)]:
        compiled=transpile(circuit,**options)
        if len(compiled.data)>200000:raise MemoryError('fixed per-component 200000 compiled-gate budget exceeded')
        rows[name]={**resource(compiled),'source_fingerprint':circuit_fingerprint(circuit)}
    write_new(DEFAULT_OUT/'components.json',{'components':rows,'compile':options,'sources':source_manifest(),
        'versions':software_versions(),'seconds':time.perf_counter()-start,'matches_current_oracle':True,
        'scope':'Four independently compiled components; not an expanded-power circuit or quantum evolution.'})
    print({name:row['gates'] for name,row in rows.items()},flush=True)


def combine():
    from rare_precision import iqae_component_cost
    c=json.loads((DEFAULT_OUT/'components.json').read_text(encoding='utf8'))
    r=json.loads((DEFAULT_OUT/'result.json').read_text(encoding='utf8'))
    assert c['sources']==r['protocol']['production_sources']==source_manifest()
    costs=iqae_component_cost(c['components'],r['estimator']['rounds'])
    assert costs['queries_A_and_inverse']==r['estimator']['costs']['queries_A_and_inverse']
    write_new(DEFAULT_OUT/'resource_cost.json',{'costs':costs,
        'scope':'Actual executed measurement rounds, including Atilde inverse and both reflections; concatenated-depth bound only.',
        'classical_reference':r['posthoc_reference'],'simulator_cpu_seconds':r['process_cpu_seconds'],
        'amplitude_cache_counts_only_as_classical_simulator_acceleration':True,
        'no_quantum_advantage_demonstrated':True})
    print(costs,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='worker':components()
    elif sys.argv[1]=='combine':combine()
    else:
        from supervised import run_limited
        r=run_limited([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],cwd=ROOT,
            output_dir=DEFAULT_OUT/'component_supervision',memory_bytes=768*1024**2,threads=1,
            heartbeat_sec=0,resource_sample_sec=1,resource_publish_sec=60,echo_output=True)
        if r['status']!='passed':raise SystemExit(2)
