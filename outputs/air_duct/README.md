# Air-duct transport

| Input preset | Grid | Source cell | Detector | Archived calculation |
|---|---|---|---|---|
| `legacy_duct` | 16 × 8 × 8, two energy groups | (1, 3, 3) | x = 12, y and z in [3, 4] | Finite-step classical recursion and a T15 circuit build |
| `q_legacy_small` | 5 × 4 × 3, one energy group | (1, 1, 1) | x = 3, y in [1, 2], z = 1 | Independent classical recursion and actual gate evolution through T5 |

Both sources start in group 0 along +x. Coordinates use zero-based indices. The input files in `configs/models` resolve through `models.load_model` and the declarations in `presets.py`. The `config` or `identity.config` field of each result contains the expanded physical parameters used for that calculation.

## Original-size duct

Open `classical_reference/legacy_duct.json` and select a finite horizon from `rows`. The columns `p_S`, `p_D`, `p_A` and `p_B` give survival, cumulative detection, absorption and escape probabilities. The unresolved survival mass bounds the additional detection possible at later steps. The T12, T15 and T22 detection probabilities are 0.4346427899325323, 0.43755881809782105 and 0.43936218225310053. At T22 the uncollided and scattered contributions are 0.3720786376356001 and 0.06728354461750041. The remaining survival mass is 0.003584468405307369.

`classical_reference/legacy_t15/result.json` contains a separate T15 regression run with its resolved input. Its first 15 detector values agree with the corresponding rows of the T1–22 run. Both records use float64 discrete recursion.

`circuit_resources/circuit_build.json` records the T15 compressed-profile circuit with 153 qubits, 9,720 top-level gates, 229 raw two-qubit gates and raw depth 2,420. The recorded stage is `built_only`. Compiled fields are null and no detector sampling was performed in this build. The cross-model summary contains its `legacy_duct` row in `../circuit_resources/resource_summary.csv`.

## Scaled duct

`gate_validation` contains three T5 files for the same scaled geometry. Each file gives the encoding in `profile`, the gate result in `actual_joint`, and the independent recursion in `reference_joint`. Each encoding has maximum joint absolute error 1.1102230246251565 × 10⁻¹⁶. The three records contribute to the 21 checks in `../validation_suite/validation_summary.json`.

## Five-step terminal channel marking

[Three-terminal marking on the reduced duct](channel_marking/baseline_T5/README.md) gives six actual sparse gate readings for the common baseline transport circuit, using the detector, absorption and leakage markers at Grover powers zero and one. The records include the independent classical probabilities, normalization, work-register checks, and discarded-amplitude bounds. A separate unfiltered run records where the basis budget was exhausted.
