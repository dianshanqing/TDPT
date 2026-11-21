# Circuit resources

For a controlled encoding comparison, open `tau23_T15/profile_counts.csv`. All three runs use the eight-cell slab `compact_n8_w0002_tau23`, 15 transport steps, the basis `u,cx`, optimization level 0, transpiler seed 917 and one transpiler process. Every two-qubit gate in this compiled basis is a CX.

| History encoding | Built qubits | Compiled gates | CX | Compiled depth |
|---|---:|---:|---:|---:|
| Baseline | 156 | 93,447 | 29,901 | 57,958 |
| Compressed | 156 | 95,472 | 30,546 | 58,486 |
| Full history | 171 | 81,537 | 26,481 | 50,564 |

In each profile folder, `circuit_build.json` captures the raw build, `compiled_resources.json` captures transpilation and `results.json` contains the final compiled fingerprint. Use `num_qubits` for the built width and `resources` for gate counts and depth. The separate `est` block gives analytical register counts. These three circuits were compiled without measurement execution.

`slab_components` extends the comparison to all three eight-cell slabs at T15. Each `components_tau*.json` gives the geometry, materials and actual compilations of baseline and compressed state preparation in the same `u,cx` basis. The additional AE component file contains compilations in the recorded Aer basis. Preparation, inverse and reflection were transpiled separately. Adding individual component depths gives a compositional upper bound. No whole-circuit amplified evolution was run in this component study.

The archived cross-model `resource_summary.csv` also records raw builds and a small MPS study. Historical evidence locators in that CSV refer to the original run tree. For the `legacy_duct` row use `../air_duct/circuit_resources`. For the other raw builds use `raw_build/<case>`. For the T16 uniform-slab transpilation use `compile_stage/uniform_tau2_w5`. The recorded transpilation finished, and the identity stage hit its memory limit. For one-cell rows use the matching bond and threshold under `../validation_suite/one_cell_mps`. The public evidence index resolves all these records from the repository root.
