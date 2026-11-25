"""Isolate scattering-event storage, scatter controls and boundary/motion controls.

Build experimental circuits from the released baseline's actual instruction stream.
Only the seven explicitly checked scatter operations can change control wires.
Usage: python -B tools/verify_history_ablation.py --outdir runs/history_new
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit, QuantumRegister, transpile

from models import config_dict, load_model
from reference_engine import evolve, marginal, observe, registers
from reference_solver import reference
from runtime_identity import circuit_fingerprint, digest, normalize, software_versions
from unitary_circuit import build_unitary_circuit_qiskit

SHORT = "configs/models/q_legacy_small.json"
SLAB = "configs/models/compact_deep/compact_n8_w0002_tau23.json"
ARCHIVE = "outputs/circuit_resources/tau23_T15"
STOCK = ("baseline", "compressed", "full_history")
VARIANTS = ("baseline_scatter_copy", "baseline_scatter_control")
ORDER = ("baseline", "compressed", *VARIANTS, "full_history")
PHYSICAL = ["status", "x", "y", "z", "dir", "g"]
TOL = 2e-11
TAIL_TOL = 2e-20
MAX_STATES = 60000
COMPILE = dict(basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=917, num_processes=1)


def write_result(path, payload):
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def build_with_boundaries(cfg, steps, profile):
    """Mark gate indices at actual returns from the unmodified production builder."""
    rows = []
    prior = sys.getprofile()
    kernel_filename = build_unitary_circuit_qiskit.__code__.co_filename

    def watch(frame, event, arg):
        name = frame.f_code.co_name
        if name not in ("compile_step_kernel", "_uncompute_in_slab_and_duct", "move_if_dir"):
            return
        if frame.f_code.co_filename != kernel_filename:
            return
        if name == "compile_step_kernel":
            if event == "call":
                rows.append(dict(step=frame.f_locals["t"], start=len(frame.f_locals["qc"].data)))
            elif event == "return":
                rows[-1]["end"] = len(frame.f_locals["qc"].data)
        elif event == "return" and frame.f_back.f_code.co_name == "compile_step_kernel":
            if name == "_uncompute_in_slab_and_duct":
                rows[-1]["movement_start"] = len(frame.f_back.f_locals["qc"].data)
            elif frame.f_locals["k"] == 5:
                rows[-1]["movement_end"] = len(frame.f_back.f_locals["qc"].data)

    sys.setprofile(watch)
    try:
        circuit, _ = build_unitary_circuit_qiskit(cfg, steps, kernel_profile=profile)
    finally:
        sys.setprofile(prior)
    assert [r["step"] for r in rows] == list(range(steps))
    assert all(r["start"] < r["movement_start"] < r["movement_end"] <= r["end"] for r in rows)
    return circuit, rows


def named_bits(circuit):
    return {q: (reg.name, i) for reg in circuit.qregs for i, q in enumerate(reg)}


def retain_scatter(base, boundaries, use_as_control):
    """Add a tS bit per step in the same register position as full_history."""
    steps = len(boundaries)
    tape = QuantumRegister(steps, "tS")
    regs = list(base.qregs)
    regs.insert(next(i for i, r in enumerate(regs) if r.name == "status"), tape)
    out = QuantumCircuit(*regs, global_phase=base.global_phase)
    names = named_bits(base)
    additions = []
    audit = []
    pending = []
    for i, inst in enumerate(base.data):
        op, wires = inst.operation, list(inst.qubits)
        first_gate = getattr(getattr(op, "base_gate", None), "label", None) == "U6"
        if first_gate:
            assert not pending
            t = len(additions)
            w = ("work", 14)
            bank = [("dirT", 3*t+j) for j in range(3)]
            direction = [("dir", j) for j in range(3)]
            assert op.num_ctrl_qubits == 1 and list(map(names.get, wires)) == [w, *bank]
            out.cx(wires[0], tape[t])
            additions.append(i)
            audit.append(dict(step=t+1, baseline_scatter_gate_index=i, added_cx=1,
                              redirected_operations=7 if use_as_control else 0))
            pending = [[w, direction[j], bank[j]] for j in range(3)]
            pending += [[w, bank[j], direction[j]] for j in range(3)]
        elif pending:
            assert op.name == "ccx" and list(map(names.get, wires)) == pending.pop(0)
        else:
            out.append(op, wires)
            continue
        if use_as_control:
            wires[0] = tape[t]
        out.append(op, wires)
    assert not pending and len(additions) == steps and len(out.data) == len(base.data) + steps
    assert all(row["start"] <= idx < row["movement_start"] for row, idx in zip(boundaries, additions))
    shifted = [{k: value if k == "step" else value+bisect_left(additions, value)
                for k, value in row.items()} for row in boundaries]
    return out, shifted, audit


def unitary_operation_rows(circuit):
    # Store ordered logical wire names, parameters (including the U6 matrix/control)
    # and controls before comparing circuits with different flat qubit numbering.
    bits = named_bits(circuit)
    for inst in circuit.data:
        op = inst.operation
        yield [op.name, getattr(op, "ctrl_state", None), list(op.params),
               [bits[q] for q in inst.qubits]]


def assemble(cfg, steps):
    builds = {p: build_with_boundaries(cfg, steps, p) for p in STOCK}
    edits = {}
    for p in VARIANTS:
        c, bounds, edits[p] = retain_scatter(*builds["baseline"], use_as_control=p==VARIANTS[1])
        builds[p] = c, bounds
    evidence = {}
    for p, (c, bounds) in builds.items():
        rows = list(unitary_operation_rows(c))
        def stream(start, end):
            return digest(rows[start:end])
        evidence[p] = dict(
            phase=float(c.global_phase), preparation_sha256=stream(0, bounds[0]["start"]),
            steps=[dict(**b, before_movement_sha256=stream(b["start"], b["movement_start"]),
                        movement_sha256=stream(b["movement_start"], b["movement_end"]),
                        after_movement_sha256=stream(b["movement_end"], b["end"])) for b in bounds])
    # Capture all gates outside the boundary/motion block in the last contrast.
    for profile in ORDER:
        assert evidence["baseline"]["phase"] == evidence[profile]["phase"]
        assert evidence["baseline"]["preparation_sha256"] == evidence[profile]["preparation_sha256"]
    for t in range(steps):
        row = lambda p: evidence[p]["steps"][t]
        assert all(row("baseline")["movement_sha256"] == row(p)["movement_sha256"]
                   for p in ("compressed", *VARIANTS))
        assert row(VARIANTS[1])["movement_sha256"] != row("full_history")["movement_sha256"]
        assert all(row(VARIANTS[1])[k] == row("full_history")[k]
                   for k in ("before_movement_sha256", "after_movement_sha256"))
    return builds, dict(gate_regions=evidence, scatter_edits=edits)


def run_short(directory):
    cfg, steps = load_model(SHORT), 3
    builds, audit = assemble(cfg, steps)
    write_result(directory / "short_gate_regions.json", audit)
    refs = [reference(cfg, t, retain_joint=True)["joint"] for t in range(1, steps+1)]
    checks = {}
    for profile in ORDER:
        circuit, bounds = builds[profile]
        points = [dict(step=b["step"], stage="step_end", gate_index=b["end"]) for b in bounds]
        def measure(qc, state, point):
            actual = marginal(qc, state, PHYSICAL)
            expected = refs[point["step"]]
            keys = sorted(set(actual) | set(expected))
            rows = [list(k)+[actual.get(k,0.), expected.get(k,0.)] for k in keys]
            err = max(abs(row[-1] - row[-2]) for row in rows)
            obs = observe(qc, state)
            obs.update(joint_rows=rows, max_joint_error=err)
            if profile in (*VARIANTS, "full_history"):
                # For these profiles tA is the interaction ledger and cA the
                # absorption indicator. Their conjunction gives scatter = tA & ~cA.
                histories = marginal(qc, state, ["tS", "tA", "cA"])
                obs["scatter_ledger_mismatch_mass"] = math.fsum(
                    v for (s,a,b), v in histories.items() if s != (a & ~b))
                obs["scatter_recorded_mass"] = math.fsum(v for (s,a,b),v in histories.items() if s)
            return obs
        state, fwd = evolve(circuit, checkpoints=points, observer=measure, max_states=MAX_STATES)
        back, reverse = evolve(circuit.inverse(), state, max_states=MAX_STATES)
        err0 = abs(back.get(0,0)-1)
        tail = math.fsum(abs(a)**2 for k,a in back.items() if k)
        step_obs = [o["observables"] for o in fwd["checkpoints"]]
        checks[profile] = len(step_obs) == steps and all(
            o["max_joint_error"] < TOL and o["work_nonzero"] < TOL and
            abs(o["norm"]-1) < TOL and o.get("scatter_ledger_mismatch_mass",0.) < TOL
            for o in step_obs) and err0 < TOL and tail < TAIL_TOL
        write_result(directory / ("short_"+profile+".json"), dict(
            profile=profile, model_input=SHORT, steps=steps, register_map=registers(circuit),
            physical_joint_columns=PHYSICAL+["actual", "independent_recurrence"],
            forward=fwd, reverse=reverse, inverse_zero_amplitude_error=err0,
            inverse_nonzero_probability=tail, passed=checks[profile]))
        print("short", profile, checks[profile], flush=True)
    return checks


def run_resource(directory):
    cfg, steps = load_model(SLAB), 15
    builds, audit = assemble(cfg, steps)
    write_result(directory / "slab_gate_regions.json", audit)
    rows = []
    fingerprints = {}
    for profile in ORDER:
        circuit, _ = builds[profile]
        compiled = transpile(circuit, **COMPILE)
        counts = dict(compiled.count_ops())
        assert set(counts) == {"u","cx"} and compiled.layout is None
        row = dict(profile=profile, qubits=circuit.num_qubits, raw_gates=circuit.size(),
                   raw_depth=circuit.depth(), compiled_u=counts["u"], compiled_cx=counts["cx"],
                   compiled_gates=compiled.size(), compiled_depth=compiled.depth())
        current_hash = circuit_fingerprint(compiled)
        identifiers = dict(compiled_fingerprint=current_hash)
        if profile in STOCK:
            old_dir = ROOT / ARCHIVE / profile
            built = json.loads((old_dir / "circuit_build.json").read_text(encoding="utf-8"))
            old = json.loads((old_dir / "compiled_resources.json").read_text(encoding="utf-8"))
            final_file = old_dir / "results.json"
            archived_hash = json.loads(final_file.read_text(encoding="utf-8"))["compiled_fingerprint"]
            identifiers.update(archived_compiled_fingerprint=archived_hash,
                               archive=final_file.relative_to(ROOT).as_posix(),
                               archive_file_sha256=hashlib.sha256(final_file.read_bytes()).hexdigest(),
                               compiled_gate_match=(archived_hash == current_hash))
            assert archived_hash == current_hash
            archive_regs = built["register_map"]
            assert normalize(config_dict(cfg)) == built["config"]
            assert registers(circuit) == archive_regs and COMPILE == old["compile_options"]
            expected = old["resources"]
            assert row["raw_depth"] == expected["raw_depth"] and row["raw_gates"] == expected["raw_gates"]
            assert row["compiled_gates"] == expected["compiled_gates"]
            assert row["compiled_cx"] == expected["compiled_two_qubit_gates"]
            assert row["compiled_depth"] == expected["compiled_depth"]
        fingerprints[profile] = identifiers
        rows.append(row)
        print("slab", row, flush=True)
    write_result(directory / "slab_compiled_fingerprints.json", fingerprints)
    with (directory / "slab_resources.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    if not __debug__:
        raise RuntimeError("Run with Python assertions enabled (omit -O).")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    directory = parser.parse_args().outdir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    files = [SHORT, SLAB, "unitary_circuit.py", "reference_engine.py", "reference_solver.py",
             "config.py", "presets.py", "models.py", "runtime_identity.py",
             Path(__file__).relative_to(ROOT).as_posix()]
    write_result(directory / "protocol.json", dict(
        reference="reference_solver.reference", simulator="reference_engine.evolve",
        cut_off="exactly zero only", max_states=MAX_STATES, strict_absolute_tolerance=TOL,
        strict_inverse_tail_tolerance=TAIL_TOL, short_input=SHORT, short_steps=3,
        short_config=config_dict(load_model(SHORT)), slab_input=SLAB, slab_steps=15,
        slab_config=config_dict(load_model(SLAB)), compile_options=COMPILE,
        coupling_map=None, backend=None, archive_resource_path=ARCHIVE, versions=software_versions(),
        sha256={p:hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}))
    checks = run_short(directory)
    if not all(checks.values()):
        raise RuntimeError("Short-circuit validation failed; see individual profile records")
    rows = run_resource(directory)
    write_result(directory / "summary.json", dict(short_checks=checks, resources=rows,
                 all_stock_archive_comparisons_passed=True, simulation_executed_on_slab=False))


if __name__ == "__main__":
    main()

