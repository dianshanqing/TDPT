# Five-step detection after three-dimensional turning

The reduced duct has grid dimensions 5 × 4 × 4. The source occupies (1,1,1) with direction +x. The detector is the single interior cell (3,2,2). Scoring occurs at the beginning of a transport step. The four moves needed to reach the window therefore give a first possible score at step 5. The [input](../../../configs/models/q_duct_offaxis.json) uses the source and the material and duct definitions of q_legacy_small. Its physical changes are the z grid extent and the detector window.

| Record | Contents |
| --- | --- |
| [Adaptive IQAE run](iqae_01/README.md) | Two sampled powers, the last valid interval, the circuit-query count and the prospective budget stop. |
| [gate_02/classical.json](gate_02/classical.json) | Independent finite-step CSR recurrence. The rows include first termination, uncollided and scattered detection, and the four terminal or surviving fractions. |
| [gate_02/preparation.json](gate_02/preparation.json) | Actual baseline preparation without amplitude screening. At all five step ends the records give both probabilities for every populated physical tuple (status, x, y, z, direction, energy group), all four status masses, normalization and the probability of any nonzero work bit. The k0 entry reports the detection flag after marking. |
| [gate_02/grover.json](gate_02/grover.json) | Marked probability after a full application of the production Grover operator to the prepared state. Each screening event lists its primitive index, state count and discarded Euclidean norm. |
| [gate_02/identity.json](gate_02/identity.json) | Input and implementation digests, transport, preparation and Grover fingerprints, register layout, software versions, and numerical settings. |
| [gate_01/grover.json](gate_01/grover.json) | Earlier run under a 650,000-entry sparse-state cap. It reached 663,552 entries at primitive 2,921 and stopped. The other files in this directory retain this attempt's input identity, classical reference and successful preparation. |
| [path_reference](path_reference/README.md) | Independent decimal enumeration of all twelve shortest direction sequences. Six have positive weight. They involve three or four changes from the incoming direction, counting the initial +x direction. |

The independent recurrence gives zero detection through step 4 and 0.00011062334505329141 at step 5. Direct sums of the squared flag-one amplitudes give 0.00011062334505329156 after preparation and 0.0009953164265524219 after one Grover iteration. The latter differs by 3.69 × 10⁻¹⁸ from the expectation computed with the classical reference. The largest physical joint-probability discrepancy across five steps is 1.11 × 10⁻¹⁶. All 17 work bits are zero at the five step ends and at both readouts within the recorded simulation. The status and detection marker agree at both readouts.

The marked circuit has 71 qubits, including the flag. The zero-state reflection covers all 71. The complete Grover evaluation traversed 5,841 primitives with an amplitude cutoff of 10⁻¹⁶ on mixing gates and peaked at 663,552 sparse entries. Its discarded Euclidean norms sum to 1.988 × 10⁻¹⁶. For retained vector norm n and discarded-norm sum ε, the bound ε(2n+ε) on any projector probability is 3.976 × 10⁻¹⁶ here. The simulation retains the full measured joint and the marker probability independently of the classical reference. This calculation tests a finite five-step response on the specified geometry. The probabilities in the gate records are simulator readouts without finite-shot sampling.

From the TDPT root in the QC environment, use a fresh output directory:

```bat
python -B tools\verify_offaxis.py --max-states 1200000 --cutoff 1e-16 --out outputs/user_runs/offaxis_gate
```

The driver exits successfully when all five joint distributions, all four status probabilities, the norm, work bits, status-marker agreement and both marked probabilities satisfy the strict 2 × 10⁻¹¹ tolerance. It compares both readouts with sin²[(2k+1) arcsin(√p)] using the full-precision independent p. The Grover acceptance checks add the screening bound to the observed discrepancies. To reproduce the earlier budget exit, set `--max-states 650000` and supply another fresh output directory.
