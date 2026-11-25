# TDPT benchmark data

Choose the physical model or the quantity you want to check. The [evidence index](evidence_index.csv) identifies the input, original result and reference for each entry. In its three path columns, each path is relative to the **project root** and a vertical bar separates multiple files. When the result and reference point to the same file, read the named reference fields alongside the measured fields in that record. An unavailable independent reference is described in the corresponding guide or index entry.

| Question | Where to start | What to inspect |
| --- | --- | --- |
| How much probability reaches the detector in the full-size air duct? | [Air duct](air_duct/README.md), `DUCT-REF-22` and `DUCT-REF-15` | Finite-step terminal probabilities, scattered and uncollided contributions, and the remaining survival mass. The small duct has its own geometry. |
| Does the reversible circuit reproduce the classical joint distribution? | [Validation suite](validation_suite/README.md), `VAL-ALL` | Seven cases times three encodings, with actual gate and reference distributions and step-end work-bit checks. `VAL-01` through `VAL-03` reside under the air duct. |
| How small a detector response is resolved on each slab grid? | [Deep penetration](deep_penetration/README.md), `DP-14-MPS`, `DP-19-MPS`, `DP-23-MPS` | Same-grid direct detector readouts, four-state diagnostics, bond/truncation controls and separate classical grid-refinement records. |
| What did the short IQAE runs measure and when did they stop? | [Amplitude estimation](amplitude_estimation/README.md), `AE-TWO-COMPLETE` and `AE-TWO-BUDGET` | Protocol, round powers, measured hits, confidence bounds and calls. Direct readout and fixed-power diagnostics have their own entries. |
| What changes when history registers are encoded differently? | [Circuit resources](circuit_resources/README.md), `RES-T15-BASELINE` through `RES-T15-FULL-HISTORY` | Matched compilation settings and actual compiled size, depth and CX counts. The full-size duct has a separate build-only record, `DUCT-BUILD-15`. |
| What can earlier calculations support? | [Historical data](historical/README.md) | Version-specific duct Monte Carlo and MPS batches, parameter scans and their validation limits. Use the batch identity and documented physical scope when comparing them with the current implementation. |

The [Light pure-absorption comparison](validation_suite/external_absorption/README.md), `EXT-ABS-AB`, supplies a continuous-flight tally, synthetic input and analytic transmission through two adjacent absorbers.

## Reproduce an experiment

Use the activated Windows prompt described in the [project README](../README.md). Run from the project root. Give each execution a new destination. The commands below use the distributed inputs, and keep new outputs under `outputs/user_runs/`.

| Calculation | Command | What to compare |
| --- | --- | --- |
| Finite-step classical references | Follow the [geometry and reference commands](../figures/fig1/README.md) for the duct and three coarse slabs. | Compare each fresh `result.json` with the indexed classical record, matching `rows` by step. The terminal channel order is recorded in `order`. |
| 21 gate validations | `call run.cmd validate --suite configs/suites/validation_small.json --cpu-budget 1 --outdir outputs/user_runs/joint21` | Open the new `validation_summary.json`; compare the per-case `actual_joint`, `reference_joint` and `runtime.checkpoints` against VAL-01 through VAL-21. The file name identifies the model and encoding. |
| Sampled transport | `call run.cmd mc --config q_uniform_small --steps 3 --histories 128 --seed 1234 --outdir outputs/user_runs/mc` | Read the four terminal counts in `results.json`, then compare their fractions to the corresponding three-step classical probabilities. The counts carry sampling uncertainty. |
| Two-shard quantum sampling | `call run.cmd test --execute --config q_uniform_small --steps 1 --shots 4 --num-shards 2 --max-parallel 2 --shot-chunk 2 --mps-max-bond 16 --run-memory-mib 3072 --outdir outputs/user_runs/gate` | Inspect `merged/results_merged.json` under the new run. To summarize the underlying shards, use `call run.cmd summarize --input outputs/user_runs/gate/shards --outdir outputs/user_runs/gate_summary`. Four shots demonstrate the execution and merge. |
| Small direct MPS readout | `call run.cmd mps_probe --config q_mps_one_cell --steps 1 --mps-max-bond 16 --mps-trunc 1e-12 --outdir outputs/user_runs/probe` | Read `mps_probability_probe.json` and the per-case JSON. Compare the detector marginal with its same-grid reference. Read `strict_status` separately from the process exit code. The distributed one-cell benchmarks use **two** steps. |
| Two-cell IQAE | Follow the [execution and count-replay commands](amplitude_estimation/README.md). | The recovered driver evolves the circuit and draws measurements. The replay reconstructs the power schedule, intervals, stopping decision and query totals from the archived counts. |

For the T15 resource table, enter this loop directly in the activated prompt. In a saved CMD file, double the percent signs. The `--probe-compile` option selects the archived `u,cx` basis, optimization level 0 and transpiler seed 917.

```bat
for %P in (baseline compressed full_history) do call run.cmd circuit --config compact_n8_w0002_tau23 --steps 15 --kernel-profile %P --probe-compile --outdir outputs/user_runs/tau23_%P
```

Compare `num_qubits` and the compiled fields of `resources` in each new `results.json` with the [three-encoding table](circuit_resources/tau23_T15/profile_counts.csv). To configure an expensive deep-penetration run, take its exact `resolved_suite.json` from the input column of DP-14-MPS, DP-19-MPS or DP-23-MPS. For example, this command previews the archived τ14 numerical settings. Remove `--dry-run` only for an actual MPS evolution. The strict four-state check failed in all three archived production runs. The [slab guide](deep_penetration/README.md) gives the observed channel errors and separate classical grid-refinement checks.

```bat
call run.cmd mps_probe --suite outputs/deep_penetration/tau14/mps/baseline_tau14_T15_bond64_01/resolved_suite.json --dry-run --outdir outputs/user_runs/tau14_full
```

The [historical guide](historical/README.md) identifies v1 sampling, summary-only records and the old solver inputs. The [Light comparison](validation_suite/external_absorption/README.md) supports recomputing the recorded tally and its analytic reference. Repeating Light's particle histories requires its external executable.

Each batch README identifies any path-redacted copies. The [source map](../provenance/source_files.csv) records the original location, original hash and distributed hash of each sourced file. Private machine locations in designated records have been replaced with repository-relative paths or named placeholders. Original run IDs and historical fingerprints refer to the recorded execution. Read input, result and reference paths in the evidence index from the project root. Use `outputs/user_runs/<new-name>/` for new calculations and `python -B tools/verify_release.py` to check the distributed files.

## Additional checks in v2.0.1

- [Three terminal channels on the reduced duct](air_duct/channel_marking/baseline_T5/README.md) compares flag measurements at two Grover powers with independent transport recursion.
- [Three-step detector oracle](amplitude_estimation/tau19_two_cell/oracle_audit_v2.0.1/README.md) reports local compilation, protocol replay and two unconditional detection probabilities classified by collision history.
- [Three-class slab recursion](deep_penetration/README.md#first-detection-by-path-class) resolves the first-detection step of uncollided, forward-scattered and direction-changing paths on the coarse grid.

## Additional comparisons in v2.0.2

- [Five-step off-axis duct](air_duct/offaxis_T5/README.md) — independent path enumeration, five gate-state checks, marker readout and budget-limited adaptive measurements.
- [Event records and movement controls](circuit_resources/history_controls/README.md) — inverse checks on three transport steps and five actual compilations of the same 15-step slab.
- [Bond-dimension sensitivity](deep_penetration/tau23/mps/baseline_tau23_T15_bond64_mem1800/README.md) — two directly comparable baseline states, all four terminal/survival probabilities, normalization and detector errors.

The source map identifies each supplementary run in the 20261125 collection and distinguishes original records from public path-redacted copies. Relative paths in the evidence index point to the corresponding public files.
