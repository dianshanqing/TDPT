# Three-step detector oracle

This v2.0.1 calculation uses the baseline two-cell slab at optical thickness 4.75. The source starts in the first slab cell. A particle must move forward twice to reach the detector in time for registration at the start of step three.

From the project root with the QC environment active, choose a new output directory for each run.

```bat
python -B tools\verify_short_oracle.py --out-dir outputs\user_runs\short_oracle_check
```

The program runs both original IQAE workers up to their circuit-identity write, intercepts that write to take the actual circuit objects, and stops each worker before sampling. It verifies the archived hashes of the driver and scientific sources, the resolved physical input and the estimator protocol for each batch. It then compiles the current circuits, checks their gates, runs one forward marked preparation, and replays the saved counts. The two historical inputs are [the completed batch](../iqae_sparse_complete_20261117_124813_443bf938/) and [the earlier budget-limited batch](../20261117_budget_limited/).

## Direct compilation

Both rows are actual Qiskit transpilation results on the local platform recorded in [result.json](result.json). The basis is `u,cx`, optimization level 0, seed 917, with unrestricted connectivity and no device mapping. The depth counts parallel layers of these basis gates.

| Circuit | Qubits | Total gates | CX | Depth |
|---|---:|---:|---:|---:|
| Marked preparation A_D | 55 | 12,150 | 4,045 | 6,023 |
| One full Grover iterate Q_D | 55 | 39,786 | 15,498 | 22,492 |

The longest recorded measurement sequence applies A_D once and Q_D 450 times. Concatenating the separately compiled blocks gives a depth upper bound of **10,127,423** for one measurement. No 450-fold circuit was constructed.

## Circuit identity and scope

The same unmarked and marked preparations match both historical hashes. Each worker rebuilt locally gives the same ordered Q_D. Its recursive fingerprint starts with `891e9f8e`, matching the earlier batch. The completed batch instead recorded `b3327223` for Q_D and a different unmarked inverse fingerprint. The archived identity records supply hashes, register width, operation counts and software versions. The completed batch has no serialized Q_D gate list for an independent comparison against the historical machine. Its marked-preparation fingerprint and the recovered source fix the intended Grover construction. The reported Q_D resources and depth bound apply to the local reconstruction.

[gate_stream.json.gz](gate_stream.json.gz) contains the entire local Q_D stream after unwrapping the outer Grover instruction. Each row gives the zero-based operation type index and ordered qubit indices, controls first. The type table gives gate names, parameters, control polarity and the complex target matrix. The wrapper phase and register map are included. The two local worker builds agree throughout. A separate check compares 3,620 gate actions against the chronological sequence of flag-Z, reversed adjoints of marked preparation, zero-state reflection on all 55 bits, and marked preparation. All controlled target matrices agree elementwise; the overall phase factor agrees with −1 to floating-point precision.

## Measurement and transport checks

The selector starts with k = 0, the upper half-circle flag set, and a [0, 1/4] interval for the angle normalized by 2π. At each round Qiskit's `_find_next_k` uses the previous k, half-circle flag and current angle interval; the estimator checks the proposed power and the new batch's query cost against their budgets before drawing fresh counts and intersecting the inverted interval with the previous one.

The completed run's powers 0, 3, 37, 450 and hits 7, 198, 219, 165 reproduce the stored interval and all 503,808 preparation/inverse calls. Each count uses a fresh binomial batch of 512 shots at its selected power. The program also regenerates the recorded counts from the stored gate probabilities and seed. Round j spends failure probability 0.05/[j(j+1)] for a two-sided Clopper–Pearson interval, totaling 0.04 in these four rounds. The estimator stops at a positive lower bound L when (U−L)/(2L) ≤ 0.01. The earlier batch requested power 668 after three rounds, above its cap of 256. A separate two-round synthetic check requests power zero twice and verifies two fresh sets of four counts, without pooling.

For the completed worker's marked forward state, the detector flag and status select all detected histories. The direction-change tape must remain zero at this horizon. The baseline collision tape `tA` distinguishes zero collisions from at least one forward scattering event. Summing squared amplitudes gives unconditional joint probabilities **0.008651695203120627** and **0.0000563332651957793**. Their sum **0.008708028468316406** differs from the archived independent detector probability by **6.94 × 10⁻¹⁸**. If f is the free-flight probability per slab cell and g is the probability of scattering into the forward direction, the two analytic references are f² and 2fg + g². The raw norm is 0.9999999999999999 and work-register leakage is zero.
