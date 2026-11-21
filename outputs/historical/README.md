# Historical transport and estimator calculations

These records preserve finite-sample counts and model comparisons. Use the result IDs beginning with `HIST-` in the [evidence index](../evidence_index.csv). The common physical model has a 16 × 8 × 8 grid, a two-cell slab, a two-by-two duct and detector window, and a source at (1, 3, 3) facing +x. The first possible detector tally occurs at step 12. The [classical recurrence](../air_duct/classical_reference/legacy_duct.json) gives the separate finite-step reference for each of the four terminal channels.

## Frozen v1 sampling records

The v1 reference implementation is identified by commit `2f326e2e33c49af3caf2b7ba29c7b8bc4d4fab2e`. The `v1_inputs` folder preserves its default configuration and the classical weighted-response driver and imports. Original quantum sampling used several circuit encodings. As part of the historical gate comparison the QPY gate order, register indices and parameters were compared with circuits rebuilt from that v1 commit using Qiskit 2.2.3 and parameter tolerance 1e-12. The original Aer version is unknown. Use v2.0.0 for new calculations.

- **duct_mps** contains finite-shot Aer MPS aggregates over step counts, bond dimensions and truncation settings. [Methods by batch](duct_mps/methods.csv) gives each archive label, actual v1 profile, bond and effective truncation label from its logs, QPY digest and measured compilation resources. [Shard counts](duct_mps/shard_counts.csv) provides the integer four-channel totals, half-open global sample intervals and raw shard SHA-256 values checked during packaging. For example, sum the four count columns for `HIST-TEST-001` to obtain 6,964 detected, 1,012 absorbed, 5,455 boundary and 1,569 surviving histories. Each sum agrees with its original `results_merged.json`. Follow the `merged_file` and `plan` columns in the methods table from the repository root. Identical file contents share a path.
- The eight runs archived under the v1 label `compressed` match the v1 **baseline** circuit. This includes the 15,000-shot T12/bond256 and T15/bond256, 384 and 512 runs. The other T12/T15 scan also contains multiple actual profiles. Use matching gate sequence hashes when comparing bond settings. Identical plans or random seeds and partially nested streams occur across runs. Combine counts only within a single result ID. The historical 18,000-shot T15/bond256 archive occurs twice.
- The `absolute_tolerance_0p01` field compares the four terminal count fractions against their respective recurrence values. Each batch uses four Bonferroni-adjusted, two-sided Clopper–Pearson intervals at 95% simultaneous confidence. A batch passes when all four intervals fit inside their target ±0.01, fails when an interval lies entirely outside, and is inconclusive otherwise. This criterion describes sampling agreement with the target at fixed MPS settings. It does not bound all sources of numerical error.
- The v1 endpoint predicate represented the two adjacent cells in each interval of this duct. It fails for singleton and wider intervals. Repeated processing after detection preserved status and position for these standard profiles but could change the detected particle's direction from T13 onward and leave residual work. All archived v1 MPS measurement layouts omitted work bits. Zero work entries in a merged v1 record denote missing measurements. Historical `default` truncation means Aer selected its historical default. These records support comparisons of the four terminal channels for this duct.
- **duct_mc** contains conventional analog Monte Carlo results for T1–T15, each with four million histories and seed 1234. The JSON records the counts, probabilities, standard errors and energy tally. All 15 four-channel count totals and ratios were checked. The original runs lack a full configuration and environment snapshot. The archived v1 default and the independent recurrence identify the comparison model. Outcomes at different steps share a seed.
- **weighted_mc** contains ten synthetic detector-efficiency experiments. For each T12 or T15 run, the detector reach probability comes from the duct. The chosen response efficiency is eta = target / reach, with targets 1e-6 through 1e-10. Each file records the reach count, the observed analog response count, and the weighted estimate eta × reach count / 1,000,000. Runs at fixed T share the transport random stream. The deep-penetration slab results use different models and detector observables.

For a small classical historical weighted-response calculation, run from the project root with a new output location.

```bat
python -B outputs\historical\v1_inputs\tools\rare_response_mc.py --steps 12 --histories 100 --seed 1234 --p-target 1e-6 --outdir outputs/user_runs/historical_weighted_example
```

The archived CMD files record the million-history settings. Their relative paths assume the original v1 layout. The original sample counts, seed scheme and driver version are part of the comparison. Choose an unused output directory for the v1 scripts; these historical drivers can replace an existing result.

To reproduce the five weighted MPS rows for HIST-TEST-046, run the postprocessor as a module from the project root. This command uses the paths in the distributed archive.

```bat
python -B -m tools.postprocess_rare_response_mps --target-csv outputs/historical/weighted_mc/rare_response_targets_s12.csv --mps-merged-json outputs/historical/duct_mps/test_046/results_merged.json --exact-json outputs/historical/exact_reference/exact_s12.json --outdir outputs/user_runs/historical_046
```

Compare the new CSV's `p_rare_Q` and `SE_rare_Q` by target with the matching batch in the [55-row archived comparison](archived_validation/archived_highbond_rare_response.csv). For HIST-TEST-047 through HIST-TEST-056, use each row's `merged_file` in [methods.csv](duct_mps/methods.csv), select the target and reference for its step count, and choose a fresh destination.

## Post-repair, pre-freeze checks

`post_repair/joint_regression/stdout.log` contains 45 JSON lines from the October 26 repaired circuit. Each line embeds the actual geometry, profile, step count, gate fingerprint, maximum discrepancy for the joint state (status, x, y, z, direction, energy) and work-bit probability. Fifteen lines also contain step-end checkpoints, totaling 54. Across the 45 cases, the largest joint discrepancy is 2.22e-16 and all final work probabilities are zero. The original log retains the comparison error for each case. The full joint reference distributions for each of these 45 cases are unavailable in this log. The formal 21-case suite has its own raw distributions.

`post_repair/independent_dp.json` contains a separate T1–T15 recurrence for the full-size duct. All four channels agree with the corresponding rows of the T1–T22 reference to within 4.6e-16.

`post_repair/t1_compressed/results.json` is a complete one-step compressed Aer run with bond 32, explicit truncation 0 and 16 simulated shots. It records 15 survivors, 1 absorption and 0 nonzero measurements across the 17 measured work qubits. Its embedded model agrees with the four physical configuration blocks in the current duct reference. Its source hashes identify the earlier repaired implementation. Its small sample size is appropriate for a measurement-format regression.

## Repeated ideal measurements

`ideal_iqae_statistics` contains 1,000 repetitions at each of five specified probabilities. Three values were taken from the same-grid slab recurrences and two were fixed at 0.2 and 0.7. The probability enters an ideal amplified-binomial count generator. Open `ideal_iqae_statistics/protocol.json` for the seed sequence, reference hash, per-round shots, alpha spending and budgets. `ideal_iqae_statistics/precision_classical.json` is the original reference selected by that hash. Each gzip file is a JSON object with a summary and per-trial observations, including incomplete runs. Counts of completed trials are 999, 1,000, 1,000, 996 and 997 in that order. The records characterize estimator performance under the specified ideal measurements. The frozen project includes the later `rare_precision.validate_relative_iqae_statistics` entry point. The protocol records the source digest of the earlier run.

`ideal_iqae_statistics/two_cell/IQAE_two_cell_budget2048.json.gz` adds 1,000 separate scalar trials with input probability 0.008708028468316413 and maximum Grover power 2048. Of these trials 997 completed and three returned an invalid interval. The summary and individual trial records permit independent aggregation of the observed results. The original batch-specific launcher for this file was not identified. For the individual transport calculations, see [amplitude estimation](../amplitude_estimation/).

## Earlier estimator and detector-response batches

- [v1_iqae](v1_iqae/) contains 15 short runs on the full-size duct, grouped by the original channel and step. Each batch README identifies the observed endpoint, the reported interval, the amount of information preserved by the library, and the classical comparison. The absolute epsilon protocol distinguishes them from later relative-precision two-cell transport runs.
- [v1_rrk](v1_rrk/) contains five statevector runs and one analytic control for a layered four-qubit reduced response kernel. The integer count tables are the input for the historical likelihood fit. The target-conditioned power plan and derived cost comparison are stored separately in the same folder. Reused target information limits these experiments to known-target calibration.
- [exact_reference](exact_reference/) contains v1 deterministic duct results through steps 3, 7, 10, 12 and 15. [archived_validation](archived_validation/) links the original conditional detector-response table to 11 finite-bond MPS settings and their detector counts. Use the evidence index to follow remapped inputs and outputs.
