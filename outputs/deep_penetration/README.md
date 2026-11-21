# Deep-penetration slabs

Each slab has eight coarse material cells, spacing 0.5, single-group scattering fraction 0.002 and total optical thickness 14, 19 or 23. The transport model uses a finite lattice, six directions and a finite time horizon. The numerical material coefficients define weakly scattering test problems. Each classical JSON contains the resolved geometry, material interaction probabilities, source and detector.

## Detector response at 15 steps

| Result ID | Optical thickness | Classical detector probability | Direct MPS probability | Relative difference | Bond cap | Strict four-state check |
|---|---:|---:|---:|---:|---:|---|
| DP-14-MPS | 14 | 8.421424906189056e-7 | 8.421342463059636e-7 | 0.00097897% | 64 | Failed |
| DP-19-MPS | 19 | 5.750336546672260e-9 | 5.750163910545263e-9 | 0.00300219% | 256 | Failed |
| DP-23-MPS | 23 | 1.072920896345993e-10 | 1.072894613808730e-10 | 0.00244962% | 256 | Failed |

These baseline circuits use double precision and a truncation threshold of 1e-16. The percentages use the classical detector probability on the identical grid and at the same step. Direct readout uses the detector marginal of the simulated state. Strict comparison applies an absolute tolerance of 2e-11 to the full four-state distribution. The tau14 record passes its fixed-model detector-channel check while the absorption and leakage discrepancies reach approximately 1.365e-6. The tau19 and tau23 records mark numerical validity as unresolved.

## First detection by path class

For the coarse 15-step slab at optical thickness 23, [tagged_tau23.py](tau23/path_classes/tagged_tau23.py) propagates independent classical live-state probabilities with three history labels at 60-digit decimal precision. [tagged_tau23.json](tau23/path_classes/tagged_tau23.json) contains all nonzero first-detection rows and the cumulative probabilities. From the project root run:

```bat
python -B outputs\deep_penetration\tau23\path_classes\tagged_tau23.py --check
```

Uncollided and forward-scattering-only paths first register at step 13. Paths that changed direction first register at steps 14 and 15. The three cumulative probabilities are 1.02618796317e-10, 4.66722647798e-12 and 6.06683959922e-15. Their sum agrees with the archived coarse-grid classical detector probability to 1.19e-16 relative. The script embeds the resolved coarse-grid parameters and compares its computed JSON with the original beside it.

## Reading the records

- Start with the three result IDs in [the evidence index](../evidence_index.csv). Each row points to the input model, resolved suite, original probability record and recurrence. A compressed JSON can be opened with Python's `gzip.open(path, "rt", encoding="utf-8")`. In the root object, `probabilities[1]` and `reference[1]` give detection. The status order is survival, detection, absorption, boundary leakage.
- Under each optical thickness, `classical/` holds independent reachable-state recurrences on the original grid and grids with one-half and one-quarter of its spacing. The last steps are 15, 29 and 57. Rows include uncollided and scattered detection and a bound on additional detection from the surviving probability. Using the finest probability as denominator, the coarse-to-fine changes are 0.684590%, 1.748521% and 3.386123%. These tests measure grid sensitivity. The high-precision file contains the separate 50- and 80-digit recurrences.
- The nine model inputs in the release are compact preset selectors. Each expands to the physical configuration saved in the corresponding classical result. Historical input-file hashes identify the earlier file representation. Use the expanded configuration to compare their physical parameters.
- Under `mps/`, each run retains its execution identity. The suite and raw result give the profile, bond cap, threshold and effective backend settings. The older `scatter_tau*` runs contain compressed-encoding controls and their small diagnostic counts. Several settings produce zero or displaced detector readouts. Keep each result attached to its own numerical settings. An identical suite or counts record shared by two runs is stored once and linked by the index.
- The `audits/` directory holds final cross-run checks. DP-19-COMP128-AUDIT and DP-23-COMP128-AUDIT identify two historical compressed runs with missing original simulator files. Their batch READMEs under `mps/` point to the retained probabilities and original-file fingerprints. DP-19-CONTROL-AUDIT and DP-RETURN-AUDIT link the later baseline controls to their final numerical assessments.
