# Event-history storage and movement control

The short test uses the released `q_legacy_small` duct for three steps. The resource comparison compiles the fifteen-step `compact_n8_w0002_tau23` slab. The models and compiled gate basis are specified in [the run protocol](qlegacy_T3_tau23_T15/protocol.json). The new profiles are experiments on the emitted baseline gate sequence. The production builder retains its original three profiles.

| Profile | Event recording and scatter control | Boundary handling before movement |
| --- | --- | --- |
| baseline | Retains the interaction and absorption records. The temporary scatter predicate controls the direction operation. | Re-evaluates the boundary predicate to clear the interior-guard contribution to the step record before the six shifts. |
| compressed | Erases the absorption coin early and further encodes the interaction record using status and direction history. Same temporary scatter control. | Same gate sequence as baseline. |
| baseline_scatter_copy | Adds one fresh `tS[t]` per step and copies the computed scatter bit into it with one CX. All original gates and their named wires are retained. | Same gate sequence as baseline. |
| baseline_scatter_control | Same copy, then uses `tS[t]` as the control of the direction unitary and the following six Toffoli gates. | Same gate sequence as baseline. |
| full_history | Same retained scatter bit and the same seven scatter controls as baseline_scatter_control. | Retains the boundary/interior record and uses it as an additional control when writing and uncomputing the movement predicate for each direction. |

All five circuits use the same direction history register, incoming source and transport parameters within each model. The last contrast changes the retained boundary record, omits its pre-movement clean-up and adds the corresponding control to each movement predicate. Its resource difference applies to those coupled operations together.

## Reproduce the observations

From the project root, activate the QC environment described in the project README. Use a new output path.

```cmd
python -B tools\verify_history_ablation.py --outdir runs\history_repeat_01
```

The driver first evolves all five three-step circuits and their inverses using the released sparse gate simulator, with exactly-zero pruning only. At each step end it checks the physical joint distribution over status, position, direction and energy group against an independent deterministic recurrence. It also checks total mass, the probability of nonzero work bits and, when present, the full scatter ledger against the retained interaction and absorption records. The inverse check starts from the computed forward state and tests return to the all-zero input, including all history and work registers. Strict thresholds are recorded in the protocol. The inverse is tested on this prepared state.

The slab run constructs all five circuits and compiles each to `u,cx` at optimization level 0 and seed 917, using one transpiler process and no coupling map or device backend. [slab_resources.csv](qlegacy_T3_tau23_T15/slab_resources.csv) gives the actual logical widths, operation counts and circuit depths. [slab_compiled_fingerprints.json](qlegacy_T3_tau23_T15/slab_compiled_fingerprints.json) gives the five compiled gate fingerprints and the three archived counterparts. At fixed baseline movement, copying fifteen scatter bits added 15 qubits, 15 CX gates and one depth layer. Substituting the recorded controls preserved gate counts and lowered depth by 645. Switching the boundary/movement handling then reduced the CX count by 3,435 and the depth by 6,750. These measurements apply to this slab and synthesis setup. The fifteen-step run ends at compilation.

## Raw records

Within `qlegacy_T3_tau23_T15/`, each `short_<profile>.json` contains the register map, measured and reference joint probabilities for every step, normalization and work checks, and forward/inverse sparse-run diagnostics. In each `*_gate_regions.json`, gate indices are zero-based half-open boundaries. SHA-256 digests cover ordered operation names, controls, parameters and named wires. They permit comparison across profiles with different flat register indices. For every step, the driver checks the baseline movement block against both added controls and compressed, and verifies that the two circuits in the final contrast agree throughout both regions outside the boundary/movement block. `summary.json` collects run outcomes and resource rows.

