"""Shared run identity, strict shard validation and protected output contracts."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent
CHANNELS = ('detect', 'absorb', 'boundary', 'survive')


def normalize(value):
    if is_dataclass(value):return normalize(asdict(value))
    if isinstance(value,dict):return {str(k):normalize(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(tuple,list)):return [normalize(v) for v in value]
    if isinstance(value,Path):return str(value)
    if isinstance(value,complex):return {'real':value.real,'imag':value.imag}
    if hasattr(value,'tolist'):return normalize(value.tolist())
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Non-finite identity value')
        return value
    if hasattr(value,'item'):return normalize(value.item())
    raise TypeError(f'Unsupported identity value: {type(value).__name__}')


def digest(value):
    return hashlib.sha256(json.dumps(normalize(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def software_versions():
    result={'python':platform.python_version(),'platform':platform.platform()}
    for package in ('qiskit','qiskit-aer','qiskit-algorithms','numpy','scipy'):
        try:result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:result[package]=None
    return result


def source_manifest(project_root=None):
    root=Path(project_root or PROJECT_ROOT).resolve()
    paths=sorted(root.glob('*.py'))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_identity(*,config,profile,compile_options,measurement,backend,phase,extra=None):
    obj=normalize({'schema_version':2,'config':config,'actual_profile':profile,
        'compile':compile_options,'measurement':measurement,'backend':backend,'phase':phase,
        'sources':source_manifest(),'versions':software_versions(),'extra':extra})
    obj['identity_hash']=digest(obj)
    return obj


def check_identity(identity):
    if not isinstance(identity,dict) or identity.get('schema_version')!=2:raise ValueError('Missing v2 identity; legacy data are not resumable')
    body={k:v for k,v in identity.items() if k!='identity_hash'}
    if identity.get('identity_hash')!=digest(body):raise ValueError('Identity checksum mismatch')
    for key in ('config','actual_profile','compile','measurement','backend','sources','versions','phase'):
        if key not in identity:raise ValueError(f'Incomplete identity: {key}')
    return identity['identity_hash']


def _resolved_output_path(path):
    """Resolve links first, then equate Windows extended and normal spellings.

    Path.resolve() may add the extended prefix only to an existing long child,
    leaving the root unprefixed. Removing that spelling difference must never
    replace actual filesystem resolution or the containment checks below.
    """
    resolved=Path(path).resolve()
    text=str(resolved)
    if sys.platform=='win32':
        if text.startswith('\\\\?\\UNC\\'):
            return Path('\\\\'+text[8:])
        if text.startswith('\\\\?\\') and len(text)>6 and text[4].isalpha() and text[5:7]==':\\':
            return Path(text[4:])
    return resolved


def validate_output_path(path,project_root=None):
    root=_resolved_output_path(project_root or PROJECT_ROOT)
    p=Path(path)
    p=_resolved_output_path(root/p if not p.is_absolute() else p)
    out=_resolved_output_path(root/'outputs')
    if out!=root/'outputs':raise ValueError('The outputs root must not redirect outside its declared directory')
    if any(part.casefold()=='pre-outputs' for part in p.parts):raise ValueError('pre-outputs is read-only')
    try:relative=p.relative_to(out)
    except ValueError as exc:raise ValueError('New output/cache must be inside project outputs/<unique-run>/') from exc
    if not relative.parts:raise ValueError('An independent output subdirectory is required')
    return p


def assert_new_file(path):
    from result_storage import assert_no_collision
    p=validate_output_path(path)
    validate_output_path(p.parent)
    assert_no_collision(p)
    return p


def prepare_run_directory(path, *, reserved_subdirectories=()):
    """Prepare a new run; an authenticated worker may retain its control folder.

    Callers must explicitly name reserved folders. Scientific files, symlinks,
    junctions and all other entries are still rejected rather than overwritten.
    """
    p=validate_output_path(path)
    if isinstance(reserved_subdirectories,str):raise ValueError('Reserved directories must be a sequence of simple names')
    reserved=set(reserved_subdirectories)
    if any(not isinstance(name,str) or not name or name in ('.','..') or '/' in name or '\\' in name
           or Path(name).is_absolute() or Path(name).drive for name in reserved):
        raise ValueError('Reserved directories must be simple names within the run')
    if p.exists() and any(child.name not in reserved or not child.is_dir() or child.is_symlink()
            or getattr(child,'is_junction',lambda:False)() for child in p.iterdir()):
        raise FileExistsError(f'Run directory is not empty: {p}; use a new run or validated runner resume')
    p.mkdir(parents=True,exist_ok=True)
    return p


def write_json_new(path,payload):
    from result_storage import write_json_new as write_stored_json
    return write_stored_json(path,payload)


def new_run_id():return uuid.uuid4().hex


def circuit_fingerprint(circuit):
    """Stream the unchanged canonical-json-v1 gate identity.

    Each instruction and nested definition is encoded exactly as before; avoid
    holding the full instruction list plus normalized/serialized copies in RAM.
    """
    from collections import OrderedDict
    # Qiskit exposes fresh lazy definitions for parameterized instructions.
    # An unbounded strong-reference memo can retain several KiB per rotation.
    # Keep strong references (id reuse must remain impossible) in a bounded LRU;
    # eviction only recomputes the same canonical digest when needed again.
    memo=OrderedDict()
    def encoded(value):
        return json.dumps(normalize(value),sort_keys=True,separators=(',',':'),
                          ensure_ascii=False,allow_nan=False).encode()
    def circuit_digest(qc):
        state=hashlib.sha256()
        meta={'qregs':[(r.name,len(r)) for r in qc.qregs],'cregs':[(r.name,len(r)) for r in qc.cregs],
              'nq':qc.num_qubits,'nc':qc.num_clbits,'global_phase':float(qc.global_phase)}
        # Sorted canonical key order, identical to the historical json.dumps.
        state.update(b'{"cregs":');state.update(encoded(meta['cregs']))
        state.update(b',"global_phase":');state.update(encoded(meta['global_phase']))
        state.update(b',"instructions":[')
        first=True
        for instruction in qc.data:
            op=instruction.operation
            record={'name':op.name,'q':[qc.find_bit(q).index for q in instruction.qubits],
                    'c':[qc.find_bit(c).index for c in instruction.clbits],
                    'params':normalize(list(op.params)),'condition':str(getattr(op,'condition',None))}
            # Gate definitions disambiguate custom operations sharing a name.
            definition=op.definition
            if definition is not None:
                key=id(definition)
                # qc.data can expose short-lived Python gate/definition wrappers.
                # Keep the definition alive: a bare id->hash cache permits Python
                # to recycle the id for a different definition later in the walk.
                if key not in memo:
                    memo[key]=(definition,circuit_digest(definition))
                    if len(memo)>256:memo.popitem(last=False)
                else:memo.move_to_end(key)
                record['definition']=memo[key][1]
            if not first:state.update(b',')
            state.update(encoded(record));first=False
        state.update(b'],"nc":');state.update(encoded(meta['nc']))
        state.update(b',"nq":');state.update(encoded(meta['nq']))
        state.update(b',"qregs":');state.update(encoded(meta['qregs']))
        state.update(b'}')
        return state.hexdigest()
    return circuit_digest(circuit)


def apply_backend_options(sim,requested):
    """Pass explicit zero through; backend rejection is fatal, never a fallback."""
    requested={k:v for k,v in requested.items() if v is not None}
    sim.set_options(**requested)
    effective={k:normalize(v) for k,v in vars(sim.options).items()}
    for key,value in requested.items():
        if key not in effective or normalize(value)!=effective[key]:raise ValueError(f'Backend did not apply {key}={value!r}')
    return {'requested':normalize(requested),'effective':effective}


def validate_cache_metadata(meta,expected_identity,circuit):
    if not isinstance(meta,dict) or meta.get('status')!='complete':raise ValueError('Cache metadata missing/incomplete')
    if check_identity(meta.get('identity'))!=check_identity(expected_identity):raise ValueError('Compiled cache identity mismatch')
    actual=circuit_fingerprint(circuit)
    if meta.get('circuit_fingerprint')!=actual:raise ValueError('Compiled QPY gate fingerprint mismatch')
    return actual


def integer(value,name):
    if type(value) is not int or value<0:raise ValueError(f'{name} must be a nonnegative integer')
    return value


def validate_completed_result(result,expected_identity=None,expected_sample=None,request_hash=None):
    if result.get('status')!='complete':raise ValueError('Result is not complete')
    h=check_identity(result.get('identity'))
    if expected_identity is not None and h!=check_identity(expected_identity):raise ValueError('Result identity mismatch')
    if request_hash is not None and result.get('request_hash')!=request_hash:raise ValueError('Result request mismatch')
    if not result.get('run_id'):raise ValueError('Missing run_id')
    sampling=result.get('sampling')
    if not isinstance(sampling,dict):raise ValueError('Missing sample provenance')
    for key in ('sample_start','sample_stop','total_samples','num_shards','shard_id','root_seed'):
        integer(sampling.get(key),key)
    n=integer(result.get('shots') if result.get('phase')=='test' else result.get('histories'),'sample count')
    if n<=0 or sampling['sample_stop']-sampling['sample_start']!=n:raise ValueError('Sample interval/count mismatch')
    if sampling['sample_stop']>sampling['total_samples']:raise ValueError('Sample interval exceeds batch')
    if sampling['num_shards']<=0 or not sampling.get('seed_strategy'):raise ValueError('Invalid shard/seed metadata')
    actual_seed=integer(sampling.get('actual_seed'),'actual seed')
    expected_seed=sampling['root_seed'] if result['phase']=='test' else sampling['root_seed']+sampling['shard_id']
    if actual_seed!=expected_seed:raise ValueError('Actual seed does not match recorded derivation rule')
    if result.get('shard_id',sampling['shard_id'])!=sampling['shard_id']:raise ValueError('Shard id fields disagree')
    if expected_sample is not None:
        for key,value in expected_sample.items():
            if sampling.get(key)!=value:raise ValueError(f'Resume sample mismatch: {key}')
    counts=result.get('counts')
    if not isinstance(counts,dict) or set(counts)!=set(CHANNELS):raise ValueError('Four raw integer terminal counts required; rounded probabilities are not counts')
    for channel in CHANNELS:
        integer(counts[channel],channel)
        legacy='n_alive' if channel=='survive' else 'n_'+channel
        if legacy in result and result[legacy]!=counts[channel]:raise ValueError('Raw terminal count fields disagree')
    if sum(counts.values())!=n:raise ValueError('Terminal counts do not sum to sample count')
    return n


def validate_result_request(result,request):
    """A self-consistent old checksum alone does not prove the requested run."""
    check_identity(result.get('identity'));identity=result['identity'];current=request['invocation']
    for key in ('config','actual_profile','sources','versions'):
        if identity.get(key)!=current.get(key):raise ValueError(f'Resume current {key} differs from completed shard')
    if identity.get('phase')!=request['phase'] or result.get('steps')!=request['steps']:
        raise ValueError('Resume phase/steps mismatch')
    expected=request.get('backend_requested')
    if expected is not None and identity.get('backend',{}).get('requested')!=expected:
        raise ValueError('Resume requested backend settings differ')
    expected_compile=request.get('compile_options')
    if expected_compile is not None and identity.get('compile')!=expected_compile:
        raise ValueError('Resume compile options differ')
    if request['phase']=='test' and 'arguments' in current:
        extra=identity.get('extra') or {}
        for key,arg in (('disable_detector','xt_disable_detector'),('disable_boundary','xt_disable_boundary')):
            if extra.get(key)!=bool(current['arguments'].get(arg,False)):
                raise ValueError(f'Resume model override differs: {arg}')


def validate_observation_metadata(results):
    first=results[0]
    for r in results:
        for key in ('steps','reg_layout','meas_order','meas_offsets','kernel_profile','backend_settings',
                    'cfg','circuit_fingerprint','circuit_signature','test_measure_x','test_measure_coins','test_measure_dir'):
            if r.get(key)!=first.get(key):raise ValueError(f'Mixed observation metadata: {key}')
        cs=r.get('coin_stats');initial=first.get('coin_stats')
        if isinstance(cs,dict)!=isinstance(initial,dict):raise ValueError('Missing coin observations in a shard')
        if isinstance(cs,dict):
            for key in ('ci1','ca1'):
                a=cs.get(key);b=initial.get(key)
                if isinstance(a,list)!=isinstance(b,list) or (isinstance(a,list) and len(a)!=len(b)):
                    raise ValueError('Coin tape dimensions differ')
                if isinstance(a,list):
                    for v in a:
                        if integer(v,'coin count')>r['shots']:raise ValueError('Coin count exceeds shots')


def validate_shard_batch(results,phase):
    if not results:raise ValueError('No shards')
    first=results[0];identity=first.get('identity');run=first.get('run_id');sampling=first.get('sampling',{})
    expected=sampling.get('num_shards')
    if len(results)!=expected:raise ValueError('Missing or extra shards')
    ids=set();intervals=[]
    for r in results:
        if r.get('phase')!=phase or r.get('run_id')!=run:raise ValueError('Different phase/run cannot be combined')
        validate_completed_result(r,expected_identity=identity,request_hash=first.get('request_hash'))
        s=r['sampling'];sid=s['shard_id']
        if sid in ids:raise ValueError('Duplicate shard id/archive')
        ids.add(sid)
        for key in ('root_seed','seed_strategy','chunk','total_samples','num_shards'):
            if s.get(key)!=sampling.get(key):raise ValueError(f'Mixed sampling metadata: {key}')
        intervals.append((s['sample_start'],s['sample_stop']))
    validate_observation_metadata(results)
    end=0
    for start,stop in sorted(intervals):
        if start!=end:raise ValueError('Overlapping/nested or missing sample interval')
        end=stop
    if end!=sampling['total_samples']:raise ValueError('Incomplete batch sample range')
    return {'run_id':run,'identity':identity,'request_hash':first.get('request_hash'),
            'sampling':{**sampling,'sample_start':0,'sample_stop':end,'shard_id':None},'status':'complete'}
