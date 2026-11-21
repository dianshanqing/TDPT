"""Reduced response kernel (RRK) for low-cost rare-response AE tests."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class RRKResult:
    p_reach_exact: float
    eta: float
    p_rare_exact: float
    layers: List[float]


def exact_probability(layers: Sequence[float], eta: float) -> RRKResult:
    p_reach = 1.0
    for q in layers:
        q = float(q)
        if not (0.0 <= q <= 1.0):
            raise ValueError("layer transmission probabilities must be in [0,1]")
        p_reach *= q
    eta = float(eta)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0,1]")
    return RRKResult(p_reach_exact=float(p_reach), eta=float(eta), p_rare_exact=float(p_reach * eta), layers=[float(q) for q in layers])


def infer_eta(layers: Sequence[float], p_target: float) -> Dict[str, Any]:
    p_reach = exact_probability(layers, 1.0).p_reach_exact
    p_target = float(p_target)
    if p_reach <= 0.0:
        return {"p_reach_exact": p_reach, "p_target": p_target, "eta": None, "feasible": False, "reason": "zero_reach"}
    eta = p_target / p_reach
    return {
        "p_reach_exact": p_reach,
        "p_target": p_target,
        "eta": float(eta),
        "feasible": bool(0.0 <= eta <= 1.0),
        "reason": "ok" if eta <= 1.0 else "eta_gt_1",
    }


def run_rrk_mc(layers: Sequence[float], eta: float, histories: int, seed: int = 1234) -> Dict[str, Any]:
    import numpy as np

    histories = int(histories)
    if histories <= 0:
        raise ValueError("histories must be positive")
    ex = exact_probability(layers, eta)
    rng = np.random.default_rng(seed)
    reach_count = 0
    analog_count = 0
    for _ in range(histories):
        reached = True
        for q in ex.layers:
            if rng.random() >= q:
                reached = False
                break
        if reached:
            reach_count += 1
            if rng.random() < eta:
                analog_count += 1
    p_reach_hat = reach_count / histories
    p_analog = analog_count / histories
    p_weighted = eta * p_reach_hat
    se_reach = math.sqrt(max(p_reach_hat * (1.0 - p_reach_hat), 0.0) / histories)
    se_analog = math.sqrt(max(p_analog * (1.0 - p_analog), 0.0) / histories)
    se_weighted = abs(eta) * se_reach
    return {
        **asdict(ex),
        "histories": histories,
        "seed": int(seed),
        "reach_count": int(reach_count),
        "analog_rare_count": int(analog_count),
        "p_reach_hat": float(p_reach_hat),
        "p_rare_analog_hat": float(p_analog),
        "p_rare_weighted_hat": float(p_weighted),
        "se_reach": float(se_reach),
        "se_rare_analog": float(se_analog),
        "se_rare_weighted": float(se_weighted),
    }


def _theta(prob: float) -> float:
    prob = min(1.0, max(0.0, float(prob)))
    return 2.0 * math.asin(math.sqrt(prob))


def ae_power_probability(p: float, m: int) -> float:
    p = min(1.0, max(0.0, float(p)))
    theta = math.asin(math.sqrt(p))
    return math.sin((2 * int(m) + 1) * theta) ** 2


def _backend_method(name: str) -> str:
    lookup = {
        "statevector": "statevector",
        "automatic": "automatic",
        "mps": "matrix_product_state",
        "matrix_product_state": "matrix_product_state",
    }
    key = str(name or "statevector").strip().lower()
    if key not in lookup:
        raise ValueError(f"unsupported RRK backend method: {name}")
    return lookup[key]


def build_rrk_state_preparation(layers: Sequence[float], eta: float):
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import RYGate

    ex = exact_probability(layers, eta)
    layer = QuantumRegister(len(ex.layers), "layer")
    obj = QuantumRegister(1, "rrk_obj")
    qc = QuantumCircuit(layer, obj, name="RRK_A")
    for i, q in enumerate(ex.layers):
        qc.ry(_theta(q), layer[i])
    if len(ex.layers) == 0:
        qc.ry(_theta(eta), obj[0])
    elif eta > 0.0:
        qc.append(RYGate(_theta(eta)).control(len(ex.layers)), list(layer) + [obj[0]])
    objective_index = qc.find_bit(obj[0]).index
    meta = {
        **asdict(ex),
        "objective_qubit": int(objective_index),
        "depth": qc.depth(),
        "qubit_count": qc.num_qubits,
        "gate_count": {str(k): int(v) for k, v in qc.count_ops().items()},
    }
    return qc, objective_index, meta


def build_rrk_grover(layers: Sequence[float], eta: float):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import GroverOperator

    A, objective_index, meta = build_rrk_state_preparation(layers, eta)
    oracle = QuantumCircuit(*A.qregs, name="RRK_O")
    oracle.z(A.qubits[objective_index])
    Q = GroverOperator(oracle=oracle, state_preparation=A)
    meta["grover_depth"] = Q.depth()
    meta["grover_gate_count"] = {str(k): int(v) for k, v in Q.count_ops().items()}
    return Q, [objective_index], meta


def _measure_objective(qc, objective_index: int):
    from qiskit import ClassicalRegister, QuantumCircuit

    try:
        base = qc.remove_final_measurements(inplace=False)
    except Exception:
        base = qc.copy()
    out = QuantumCircuit(*base.qregs)
    out.compose(base, inplace=True)
    c = ClassicalRegister(1, "cobj")
    out.add_register(c)
    out.measure(out.qubits[int(objective_index)], c[0])
    return out


def _build_power_circuit(Q, m: int):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(*Q.state_preparation.qregs, name=f"RRK_Qpow_{int(m)}_A")
    qc.compose(Q.state_preparation, inplace=True)
    for _ in range(int(m)):
        qc.compose(Q, inplace=True)
    return qc


def run_rrk_fixed_depth(
    layers: Sequence[float],
    eta: float,
    powers: Sequence[int] = (0, 1, 2),
    shots: int = 2000,
    seed: int = 1234,
    backend_method: str = "statevector",
    backend_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    t0 = time.time()
    exact = exact_probability(layers, eta)
    Q, objective_qubits, meta = build_rrk_grover(layers, eta)
    objective = int(objective_qubits[0])
    method = _backend_method(backend_method)
    sim = AerSimulator(method=method, seed_simulator=int(seed))
    if backend_options:
        sim.set_options(**backend_options)
    rows: List[Dict[str, Any]] = []
    for m in [int(x) for x in powers]:
        build0 = time.time()
        qc = _build_power_circuit(Q, m)
        qcm = _measure_objective(qc, objective)
        try:
            tqc = transpile(qcm, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
        except Exception:
            tqc = transpile(qcm, optimization_level=0)
        build_elapsed = time.time() - build0
        run0 = time.time()
        counts = sim.run(tqc, shots=int(shots), seed_simulator=int(seed) + 1000 * int(m)).result().get_counts()
        run_elapsed = time.time() - run0
        success = int(counts.get("1", 0))
        fail = int(counts.get("0", 0))
        total = success + fail
        p_success = success / total if total else 0.0
        rows.append(
            {
                "sampler_type": "quantum_circuit",
                "backend_method": method,
                "m": int(m),
                "oracle_calls": int(2 * m + 1),
                "shots": int(total),
                "success_count": int(success),
                "failure_count": int(fail),
                "success_probability": float(p_success),
                "expected_success_probability": float(ae_power_probability(exact.p_rare_exact, m)),
                "success_se": math.sqrt(max(p_success * (1.0 - p_success), 0.0) / total) if total else 0.0,
                "circuit_depth": tqc.depth(),
                "gate_count": {str(k): int(v) for k, v in tqc.count_ops().items()},
                "qubit_count": tqc.num_qubits,
                "build_time_sec": float(build_elapsed),
                "elapsed_time_sec": float(run_elapsed),
            }
        )
    return {
        "phase": "rrk_fixed_depth_ae",
        "sampler_type": "quantum_circuit",
        "backend_method": method,
        "backend_options": backend_options or {},
        "exact": asdict(exact),
        "powers": [int(x) for x in powers],
        "shots": int(shots),
        "seed": int(seed),
        "rows": rows,
        "meta": meta,
        "total_wall_time_sec": float(time.time() - t0),
    }


def run_rrk_analytic_fixed_depth(
    layers: Sequence[float],
    eta: float,
    powers: Sequence[int] = (0, 1, 2),
    shots: int = 2000,
    seed: int = 1234,
) -> Dict[str, Any]:
    import numpy as np

    t0 = time.time()
    exact = exact_probability(layers, eta)
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    qubit_count = len(exact.layers) + 1
    for m in [int(x) for x in powers]:
        run0 = time.time()
        expected = ae_power_probability(exact.p_rare_exact, m)
        success = int(rng.binomial(int(shots), expected))
        fail = int(shots) - success
        p_success = success / int(shots) if int(shots) else 0.0
        rows.append(
            {
                "sampler_type": "analytic",
                "backend_method": "analytic",
                "m": int(m),
                "oracle_calls": int(2 * m + 1),
                "shots": int(shots),
                "success_count": int(success),
                "failure_count": int(fail),
                "success_probability": float(p_success),
                "expected_success_probability": float(expected),
                "success_se": math.sqrt(max(p_success * (1.0 - p_success), 0.0) / int(shots)) if int(shots) else 0.0,
                "circuit_depth": "",
                "gate_count": {},
                "qubit_count": int(qubit_count),
                "build_time_sec": 0.0,
                "elapsed_time_sec": float(time.time() - run0),
            }
        )
    return {
        "phase": "rrk_fixed_depth_ae",
        "sampler_type": "analytic",
        "backend_method": "analytic",
        "exact": asdict(exact),
        "powers": [int(x) for x in powers],
        "shots": int(shots),
        "seed": int(seed),
        "rows": rows,
        "meta": {
            **asdict(exact),
            "objective_qubit": int(qubit_count - 1),
            "depth": "",
            "qubit_count": int(qubit_count),
            "gate_count": {},
            "note": "Analytic binomial sampler; no quantum circuit was built.",
        },
        "total_wall_time_sec": float(time.time() - t0),
    }
