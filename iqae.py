"""IQAE and Aer sampler helpers for transport amplitude-estimation checks."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Optional, Any, Sequence, Tuple, Dict, List

def _ts(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def _import_qiskit():
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import GroverOperator
    return (QuantumCircuit, QuantumRegister, GroverOperator)

def _import_qiskit_algorithms():
    try:
        from qiskit_algorithms.amplitude_estimators import IterativeAmplitudeEstimation, EstimationProblem
        return (IterativeAmplitudeEstimation, EstimationProblem)
    except Exception:
        from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
        return (IterativeAmplitudeEstimation, EstimationProblem)

def _import_aer():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator
    except Exception as e:
        raise ImportError(f'Could not import qiskit_aer.AerSimulator. Please install qiskit-aer.\n{repr(e)}') from e

@dataclass
class OracleSpec:
    good: str = 'detect'
    target_g: Optional[int] = None
    rare_eta: Optional[float] = None
    p_target: Optional[float] = None
_GOOD_TO_STATUS = {'survive': 0, 'detect': 1, 'absorb': 2, 'boundary': 3}

def _is_rare_detect_good(good: str) -> bool:
    return str(good).strip().lower().replace("-", "_") in ("rare_detect", "rare_detect_response", "rare_response_detect")

def _ops_dict(qc) -> Dict[str, int]:
    try:
        return {str(k): int(v) for k, v in qc.count_ops().items()}
    except Exception:
        return {}

def _circuit_stats(qc) -> Dict[str, Any]:
    return {
        "num_qubits": int(getattr(qc, "num_qubits", 0) or 0),
        "depth": None if qc is None else qc.depth(),
        "size": None if qc is None else qc.size(),
        "gate_count": _ops_dict(qc),
    }

def rare_response_angle(eta: float) -> float:
    eta = float(eta)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0, 1]")
    return 2.0 * math.asin(math.sqrt(eta))

def _as_single_qubit(qubit_like):
    try:
        if len(qubit_like) == 1:
            return qubit_like[0]
    except TypeError:
        pass
    return qubit_like

def build_rare_detect_marker(circuit, status_register, objective_qubit, eta: float):
    """Append a detect-conditioned rare-response coin to an existing circuit.

    The transport circuit remains unchanged.  The marker applies RY(theta_eta)
    to objective_qubit only on terminal status 01=detect, so objective=1 has
    conditional probability eta when the objective qubit starts in |0>.
    """
    from qiskit.circuit.library import RYGate

    theta = rare_response_angle(eta)
    if len(status_register) < 2:
        raise ValueError("status_register must contain at least two qubits")
    s0, s1 = status_register[0], status_register[1]
    obj = _as_single_qubit(objective_qubit)

    if theta == 0.0:
        return {"eta": float(eta), "theta_eta": float(theta), "status_code": "01"}

    circuit.x(s1)
    circuit.append(RYGate(theta).control(2), [s0, s1, obj])
    circuit.x(s1)
    return {"eta": float(eta), "theta_eta": float(theta), "status_code": "01"}

def build_grover_operator_with_rare_detect_marker(*, A_circuit, status_qubits, eta: float):
    QuantumCircuit, QuantumRegister, GroverOperator = _import_qiskit()
    eta = float(eta)
    rare_response_angle(eta)
    objective = QuantumRegister(1, "rare_obj")
    full = QuantumCircuit(*A_circuit.qregs, objective, name="A_rare_detect")
    full.compose(A_circuit, inplace=True)
    marker_meta = build_rare_detect_marker(full, status_qubits, objective, eta)
    oracle = QuantumCircuit(*A_circuit.qregs, objective, name="O_rare_obj_Z")
    oracle.z(objective[0])
    if oracle.num_qubits != full.num_qubits:
        raise RuntimeError(f"oracle.num_qubits({oracle.num_qubits}) != state_preparation.num_qubits({full.num_qubits})")
    grover = GroverOperator(oracle=oracle, state_preparation=full, reflection_qubits=list(range(full.num_qubits)))
    obj_idx = full.find_bit(objective[0]).index
    return (grover, [obj_idx], marker_meta)

def build_grover_operator_with_objective_flag(*, A_circuit, status_qubits, g_qubits, spec: OracleSpec, mcx_mode: str='noancilla'):
    QuantumCircuit, QuantumRegister, GroverOperator = _import_qiskit()
    good_key = str(spec.good).strip().lower()
    if good_key not in _GOOD_TO_STATUS:
        raise ValueError(f"Unknown good='{spec.good}'. Expected one of {list(_GOOD_TO_STATUS.keys())}.")
    desired = _GOOD_TO_STATUS[good_key]
    flag = QuantumRegister(1, 'flag')
    full = QuantumCircuit(*A_circuit.qregs, flag, name='A_full')
    full.compose(A_circuit, inplace=True)
    s0, s1 = (status_qubits[0], status_qubits[1])
    if desired >> 0 & 1 == 0:
        full.x(s0)
    if desired >> 1 & 1 == 0:
        full.x(s1)
    controls: List[Any] = [s0, s1]
    if spec.target_g is not None:
        from ae_likelihood import integer
        tg = integer(spec.target_g, 'target_g')
        if tg < 0 or tg >= 1 << len(g_qubits):
            raise ValueError(f'target_g={tg} is outside the {len(g_qubits)}-bit energy register')
        for i in range(len(g_qubits)):
            if tg >> i & 1 == 0:
                full.x(g_qubits[i])
        controls = controls + list(g_qubits)
    full.mcx(controls, flag[0], mode=mcx_mode)
    if spec.target_g is not None:
        from ae_likelihood import integer
        tg = integer(spec.target_g, 'target_g')
        for i in range(len(g_qubits)):
            if tg >> i & 1 == 0:
                full.x(g_qubits[i])
    if desired >> 0 & 1 == 0:
        full.x(s0)
    if desired >> 1 & 1 == 0:
        full.x(s1)
    oracle = QuantumCircuit(*A_circuit.qregs, flag, name='O_flagZ')
    oracle.z(flag[0])
    if oracle.num_qubits != full.num_qubits:
        raise RuntimeError(f'oracle.num_qubits({oracle.num_qubits}) != state_preparation.num_qubits({full.num_qubits})')
    grover = GroverOperator(oracle=oracle, state_preparation=full, reflection_qubits=list(range(full.num_qubits)))
    obj_idx = full.find_bit(flag[0]).index
    return (grover, [obj_idx])

class _CountsLike:

    def __init__(self, counts: Dict[str, int]):
        self._counts = dict(counts)

    def get_counts(self, *args, **kwargs) -> Dict[str, int]:
        return dict(self._counts)

class _SamplerData:

    def __init__(self, key: str, value_obj: Any):
        self._key = key
        setattr(self, key, value_obj)

    def keys(self):
        return [self._key]

class _SamplerResultItem:

    def __init__(self, data: _SamplerData, metadata: dict):
        self.data = data
        self.metadata = metadata

class _JobLike:

    def __init__(self, payload_list: list):
        self._payload_list = payload_list

    def result(self):
        return self._payload_list

class _PubsToAerObjectiveSampler:

    def __init__(self, shots: int, objective_qubit_index: int, seed: Optional[int]=None, verbose: bool=False, backend_options: Optional[dict]=None, debug: bool=False, heartbeat_sec: float=30.0, shot_chunk: int=100, print_ops: bool=False, max_call_seconds: Optional[float]=None, progress_state: Optional[dict]=None):
        self._shots = int(shots)
        self._obj = int(objective_qubit_index)
        self._seed = seed
        self._verbose = bool(verbose)
        self._debug = bool(debug)
        heartbeat_sec = 30.0 if heartbeat_sec is None else float(heartbeat_sec)
        self._heartbeat_sec = heartbeat_sec if heartbeat_sec > 0 else 0.0
        self._shot_chunk = int(shot_chunk) if int(shot_chunk) > 0 else 0
        self._print_ops = bool(print_ops)
        self._max_call_seconds = None if max_call_seconds is None else float(max_call_seconds)
        if self._max_call_seconds is not None and (not math.isfinite(self._max_call_seconds) or self._max_call_seconds <= 0):
            raise ValueError('max_call_seconds must be None or positive and finite')
        self._progress_state = progress_state if isinstance(progress_state, dict) else None
        self._call_id = 0
        AerSimulator = _import_aer()
        if seed is None:
            self._backend = AerSimulator(method='matrix_product_state')
        else:
            self._backend = AerSimulator(method='matrix_product_state', seed_simulator=int(seed))
        self._backend_options = {"max_parallel_threads": 1, **(backend_options or {})}
        for key in ('max_parallel_threads', 'mps_omp_threads'):
            if key in self._backend_options and (type(self._backend_options[key]) is not int or self._backend_options[key] <= 0):
                raise ValueError(f'{key} must be a positive integer')
        if self._shots <= 0:
            raise ValueError("shots must be positive")
        from runtime_identity import apply_backend_options
        self.effective_backend_options = apply_backend_options(self._backend, self._backend_options)
        self.active_power = 0
        from qiskit import QuantumCircuit, ClassicalRegister, transpile
        self._QuantumCircuit = QuantumCircuit
        self._ClassicalRegister = ClassicalRegister
        self._transpile = transpile
        self._safe_basis_gates = ['rz', 'sx', 'x', 'cx']

    def _qc_stats(self, qc, topk: int=10) -> dict:
        try:
            size = qc.size()
        except Exception:
            try:
                size = len(qc.data)
            except Exception:
                size = None
        try:
            depth = qc.depth()
        except Exception:
            depth = None
        try:
            name = getattr(qc, 'name', None)
        except Exception:
            name = None
        ops_top = None
        if self._print_ops:
            try:
                ops = qc.count_ops()
                ops_top = sorted([(str(k), int(v)) for k, v in ops.items()], key=lambda x: (-x[1], x[0]))[:topk]
            except Exception:
                ops_top = None
        return {'name': name, 'num_qubits': getattr(qc, 'num_qubits', None), 'depth': depth, 'size': size, 'ops_top': ops_top}

    def _fmt_ops_top(self, ops_top) -> str:
        if not ops_top:
            return ''
        return ' | top_ops=' + ', '.join([f'{k}={v}' for k, v in ops_top])

    def _compile_for_aer(self, qc, call_id: int, pub_id: int):
        # One deterministic compilation attempt. Failures propagate to the journal.
        tqc = self._transpile(qc, basis_gates=self._safe_basis_gates,
                              optimization_level=0, seed_transpiler=self._seed)
        if list(tqc.parameters):
            raise RuntimeError(f"IQAE compile left unbound parameters: {list(tqc.parameters)}")
        allowed = set(self._safe_basis_gates + ['measure', 'barrier', 'id'])
        bad = sorted(set(tqc.count_ops()) - allowed)
        if bad:
            raise RuntimeError(f"IQAE compile contains unsupported operations: {bad}")
        return tqc

    def _force_measure_objective_only(self, qc):
        qc0 = qc.remove_final_measurements(inplace=False)
        qcm = self._QuantumCircuit(*qc0.qregs)
        qcm.compose(qc0, inplace=True)
        cobj = self._ClassicalRegister(1, 'cobj')
        qcm.add_register(cobj)
        qcm.measure(qcm.qubits[self._obj], cobj[0])
        return qcm

    def run(self, pubs: Sequence[Tuple[Any, ...]]):
        from ae_execution import run_sampler
        return run_sampler(self, pubs)


def run_iqae_small(**kwargs):
    """Supervised IQAE; incomplete budgets return (None, metadata)."""
    from ae_runtime import run_ae_worker
    output_dir = kwargs.pop("output_dir", None)
    memory_bytes = kwargs.pop("memory_bytes", None)
    timeout = kwargs.get("sampler_max_call_seconds")
    timeout = None if timeout is None else float(timeout)
    if kwargs.get("iqae_max_grover_power") is None:
        kwargs["iqae_max_grover_power"] = 16
    result = run_ae_worker("iqae", kwargs, timeout_sec=timeout, output_dir=output_dir, memory_bytes=memory_bytes)
    return result.get("value"), result["meta"]


def _run_iqae_in_process(**kwargs):
    from ae_execution import run_iqae
    return run_iqae(**kwargs)


def build_grover_operator_for_spec(*, A_circuit, status_qubits, g_qubits, spec: OracleSpec):
    if _is_rare_detect_good(spec.good):
        if spec.target_g is not None:
            raise ValueError("rare_detect fixed-depth AE currently marks detect-only response and does not support target_g")
        if spec.rare_eta is None:
            raise ValueError("rare_detect fixed-depth AE requires OracleSpec.rare_eta")
        Q, objective_qubits, marker_meta = build_grover_operator_with_rare_detect_marker(
            A_circuit=A_circuit,
            status_qubits=status_qubits,
            eta=float(spec.rare_eta),
        )
        return Q, objective_qubits, marker_meta
    Q, objective_qubits = build_grover_operator_with_objective_flag(
        A_circuit=A_circuit,
        status_qubits=status_qubits,
        g_qubits=g_qubits,
        spec=spec,
        mcx_mode='noancilla',
    )
    return Q, objective_qubits, None

def _build_power_circuit(Q, m: int):
    QuantumCircuit, _, _ = _import_qiskit()
    qc = QuantumCircuit(*Q.state_preparation.qregs, name=f"Qpow_{int(m)}_A")
    qc.compose(Q.state_preparation, inplace=True)
    for _ in range(int(m)):
        qc.compose(Q, inplace=True)
    return qc

def run_fixed_depth_ae_sampling(**kwargs):
    """Supervised fixed-depth sampling with an explicitly bounded schedule."""
    from ae_runtime import run_ae_worker
    output_dir = kwargs.pop("output_dir", None)
    memory_bytes = kwargs.pop("memory_bytes", None)
    timeout = kwargs.get("sampler_max_call_seconds")
    timeout = None if timeout is None else float(timeout)
    if kwargs.get("iqae_max_grover_power") is None:
        kwargs["iqae_max_grover_power"] = 16
    result = run_ae_worker("fixed", kwargs, timeout_sec=timeout, output_dir=output_dir, memory_bytes=memory_bytes)
    result.setdefault("rows", [])
    return result


def _run_fixed_in_process(**kwargs):
    from ae_execution import run_fixed
    return run_fixed(**kwargs)


def relative_interval_stop(interval, relative_error=.01):
    """Midpoint guarantee conditional on coverage; no reference value is read."""
    lo, hi = map(float, interval)
    if not math.isfinite(relative_error) or not 0 < relative_error < 1:
        raise ValueError('relative_error must be in (0,1)')
    if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo <= hi <= 1:
        raise ValueError('invalid confidence interval')
    return lo > 0 and (hi-lo)/(2*lo) <= relative_error


def estimate_relative_iqae(sample_counts, *, relative_error=.01, confidence=.95,
                           shots=512, max_shots=200000, max_queries=100000000000,
                           max_power=10000000):
    """IQAE inversion with anytime-valid failure spending and resource budgets.

    sample_counts(k,n) supplies a fresh Binomial(n, sin²((2k+1)theta)) count.
    It receives no reference probability. We reuse pinned Qiskit's IQAE power
    selector, but use fresh batches and alpha_t=alpha/[t(t+1)]. Conditional on
    all earlier batches, exact CP coverage fails with probability <=alpha_t.
    A union bound over all rounds is <=alpha, also under adaptive stopping.
    On that event the intersected angle interval contains theta. For p in
    [L,U], midpoint error / p <= (U-L)/(2L). This is a statistical guarantee
    for the ideal measurement law, not a finite-bond simulator certification.
    """
    from scipy.stats import beta
    from numbers import Integral
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be in (0,1)')
    relative_interval_stop([0.,1.], relative_error)
    for name, value, minimum in [('shots',shots,1),('max_shots',max_shots,1),
                                 ('max_queries',max_queries,1),('max_power',max_power,0)]:
        if isinstance(value,bool) or not isinstance(value,Integral) or value < minimum:
            raise ValueError(name+' has invalid integer budget')
    IAE, _ = _import_qiskit_algorithms()
    selector = IAE(epsilon_target=.1, alpha=1-confidence)
    theta = [0., .25]; interval = [0., 1.]; power = 0; upper = True
    rows = []; used_shots = 0; queries = 0; grover = 0
    status = 'budget_limited'; reason = '预算内未达到目标'; estimate = None
    while used_shots < max_shots:
        candidate, branch = selector._find_next_k(power, upper, theta)
        if candidate > max_power:
            reason = '预算内未达到目标: maximum Grover power'; break
        count = min(shots, max_shots-used_shots)
        if queries + count*(2*candidate+1) > max_queries:
            reason = '预算内未达到目标: A/A-inverse query budget'; break
        power, upper = int(candidate), bool(branch)
        t = len(rows)+1; alpha_t = (1-confidence)/(t*(t+1))
        try:
            hits = sample_counts(power, int(count))
            if isinstance(hits,bool) or not isinstance(hits,Integral) or not 0 <= hits <= count:
                raise ValueError('sampler must return an integer success count in [0,shots]')
            hits = int(hits)
        except Exception as exc:
            status = getattr(exc,'status','failed'); reason = repr(exc); break
        used_shots += count; queries += count*(2*power+1); grover += count*power
        # Fresh batch only: pooling across adaptively repeated powers would
        # need a different confidence sequence proof.
        low = 0. if hits == 0 else float(beta.ppf(alpha_t/2,hits,count-hits+1))
        high = 1. if hits == count else float(beta.isf(alpha_t/2,hits+1,count-hits))
        if not all(math.isfinite(v) for v in (low, high)) or not 0 <= low <= high <= 1:
            rows.append({'round':t,'power':power,'shots':count,'success_count':hits,
                         'alpha':alpha_t,'invalid_measurement_interval':list(map(repr,(low,high)))})
            status='invalid_interval'; reason='non-finite or unordered binomial interval'; break
        phi_l = math.asin(math.sqrt(low))/math.pi
        phi_u = math.asin(math.sqrt(high))/math.pi
        if not upper: phi_l, phi_u = 1-phi_u, 1-phi_l
        scaling = 4*power+2
        new_theta = [(math.floor(scaling*theta[0])+phi_l)/scaling,
                     (math.floor(scaling*theta[1])+phi_u)/scaling]
        new_theta = [max(theta[0],new_theta[0]), min(theta[1],new_theta[1])]
        row = {'round':t,'power':power,'shots':count,'success_count':hits,
               'alpha':alpha_t,'measurement_interval':[low,high],
               'theta_interval':new_theta,'prior_theta_interval':theta}
        rows.append(row)
        if not all(math.isfinite(v) for v in new_theta) or not 0 <= new_theta[0] <= new_theta[1] <= .25:
            status='invalid_interval'; reason='empty or invalid angle interval'; break
        theta = new_theta
        interval = [math.sin(2*math.pi*v)**2 for v in theta]
        row['confidence_interval'] = interval
        if relative_interval_stop(interval,relative_error):
            status='completed'; reason='relative midpoint criterion'; estimate=sum(interval)/2; break
    success=status=='completed'
    return {'status':status,'reason':reason,'estimate':estimate,
            'confidence_interval':interval if success else None,'last_valid_interval':interval,
            'precision_target_met':success,'relative_error':relative_error,'confidence':confidence,'alpha':1-confidence,
            'confidence_scope':'single complete estimation process',
            'failure_probability_spending':'alpha_t=(1-confidence)/(t*(t+1)); fresh batches',
            'failure_probability_spent':sum(r['alpha'] for r in rows),
            'stopping_rule':'L>0 and (U-L)/(2L)<=relative_error; midpoint estimate',
            'statistical_precision':'certified_under_ideal_measurement_law' if success else 'not_achieved',
            'simulation_error':'not_bounded_by_estimator','full_pipeline_certified':False,
            'budgets':{'shots_per_round':shots,'max_shots':max_shots,'max_queries':max_queries,
                       'max_power':max_power,'wall_clock_deadline':None},
            'costs':{'shots':used_shots,'queries_A_and_inverse':queries,'grover_iterations':grover,
                     'A_calls':used_shots+grover,'A_inverse_calls':grover,
                     'zero_reflection_calls':grover,'objective_reflection_calls':grover,
                     'max_power':max((r['power'] for r in rows),default=0)},'rounds':rows}
