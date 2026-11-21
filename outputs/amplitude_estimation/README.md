# Short transport amplitude estimation

The objective is detection in a finite transport circuit. The estimator allocates fresh measurements at power k of the Grover operator and inverts the observed binomial counts. For a positive lower endpoint L and upper endpoint U, the 1% stopping rule is (U−L)/(2L) ≤ 0.01. Confidence 0.95 refers to one complete adaptive estimation procedure under the ideal measurement law. Simulator error and spatial discretization require separate checks.

## Two-cell, three-step sparse runs

| Result ID | Seed | Powers | Success counts, each out of 512 | Shots | A and inverse calls | Relative bound | Status |
|---|---:|---|---|---:|---:|---:|---|
| AE-TWO-COMPLETE | 2026111701 | 0, 3, 37, 450 | 7, 198, 219, 165 | 2,048 | 503,808 | 0.160570% | Target met |
| AE-TWO-BUDGET | 2026111001 | 0, 5, 56 | 3, 361, 419 | 1,536 | 64,000 | 1.231991% | Maximum-power budget reached |

Read `tau19_two_cell/iqae_sparse_complete_20261117_124813_443bf938/protocol.json` for the source hashes, model, seed, budgets and version information. The completed batch supplements the v2.0.0 freeze. Its protocol specifies independent fresh counts. The matching result contains the raw per-power measurements, the confidence interval after every round and the independent recurrence used after termination. The reference detector probability is 0.008708028468316413. The completed estimate is 0.008711676928702730 and its interval is [0.008697710986023482, 0.008725642871381981]. The query total combines 252,928 preparations and 250,880 inverse preparations. For n shots at power k the contribution is n(2k+1).

All 44 frozen root Python sources match the production hashes in the completed run. Its protocol names an additional driver, `iqae_sparse_complete.py`, with SHA-256 `4abb0473ab9f10b05f2fa5247e4029cd8fd17262e761313a7a23e34d79e40956`. The matching file was unavailable for this archive. The saved power-by-power probabilities and post hoc reference support comparison with the recorded counts. Run the following lightweight checks from the project root in the QC environment to reproduce both estimators from those counts.

```bat
python -B tools\replay_short_iqae.py outputs\amplitude_estimation\tau19_two_cell\iqae_sparse_complete_20261117_124813_443bf938\result.json
python -B tools\replay_short_iqae.py outputs\amplitude_estimation\tau19_two_cell\20261117_budget_limited\result.json
```

In `20261117_budget_limited`, `driver_executed.py` matches the earlier protocol fingerprint. The two scripts in `component_drivers/` supplied that batch's separate component-cost calculation. Their historical defaults refer to the source layout used at the time. They are archived source records. Four components were independently compiled. Multiplying their costs by the recorded calls gives estimated full-run gate counts and depth bounds.

## Other records

- `short_case_selection/` records two independent recurrence references and their resolved configurations. The sparse runs in `tau19_one_cell/20261107/` and `q_uniform_small/20261107/` reached the target at two and three steps, respectively. The original driver is the `iqae` stage of `tests/collect_freeze_evidence.py`.
- `q_mps_one_cell/` contains the two-step unpruned sparse-gate run and the measured MPS implementation. Their estimates and shot totals coincide. `q_uniform_small/uniform_T3_b128_01/` contains the separate three-step MPS run. The worker results and attempt logs preserve completed counts, power schedules, backend settings and query costs. For the MPS estimator, inspect `meta.statistical_costs.queries_A_and_inverse`.
- `tau19_two_cell/tau19_cells2_T3_b128_01/` records an earlier MPS attempt. It returned 3 and 368 successes at powers 0 and 5, each from 512 shots. Compilation failed at power 64. The 1,024 measurements belong to this batch and were excluded from both sparse runs.
- `fixed_power/` contains separate diagnostic runs. The later power-64 record reached compilation with zero backend submissions. The two probability runs measured fixed-power marginals. The audit table includes observed work-register residuals and marks numerical validity unresolved for both fixed-power comparisons. The raw paths in the table identify the source layout. For paths inside TDPT, use [the evidence index](../evidence_index.csv).
