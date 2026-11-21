# Terminal markers on the reduced air duct

The input [q_legacy_small.json](../../../../configs/models/q_legacy_small.json) selects the 5 × 4 × 3 duct and its source. All three measurements use the same five-step baseline transport circuit. The terminal marker tests the status register for detection (01), absorption (10), or boundary leakage (11). The additional flag is the only measurement target. The zero-state reflection acts on every qubit of the marked preparation, including the flag and history.

## Read the records

* `result.json` contains the six rows under `measurements`. Each `observed_marked_probability` is the sum of squared complex amplitudes with the flag equal to one, after the marked preparation (k=0) or one actual Grover iteration (k=1). The independent classical recurrence supplies the input to the expected value sin²[(2k+1) arcsin(√p)]. The sine can decrease between successive powers.
* `cutoff_comparison.json` repeats all six measurements with a different numerical cutoff. `unfiltered_detect_budget.json` records a separate unfiltered detection run: it completed k=0, then reached 414,729 sparse entries at the 2,137th primitive in Q. This exceeded the 300,000-entry budget.
* Each record includes the resolved model, software versions, hashes of the input and implementation, the common transport circuit and each marked preparation and Grover circuit. The baseline gate-validation [record](../../gate_validation/q_legacy_small_baseline.json) gives the independently archived unmarked circuit fingerprint and full transport marginals.
* The transport, classical recurrence and gate application are from the v2.0.0 scientific baseline. The records here were generated for the v2.0.1 release by the [terminal-marker validation driver](../../../../tools/verify_transport_channels.py).

For k=0 and k=1 respectively, the observed flag probabilities are 0.2782939560903015 and 0.9907558352824102 for detection, 0.5097997911364661 and 0.47061568469977827 for absorption, and 0.21178623774583738 and 0.9815836487005388 for boundary leakage. The maximum absolute difference from the six expected values is 3.33e-16. The maximum filtering contribution bound is 5.58e-16.

## Numerical method

The driver reuses the sparse gate engine and the production marker/Grover constructors. It evolves the marked preparation without a cutoff and checks the norm and the joint probability of any nonzero work bit at all five transport step ends. For k=1, it applies the actual Grover primitives, including their controls, complex matrices and phases. The simulator implements each unconditional X by an exact XOR relabeling of sparse basis indices. The [one-step test](../../../../tests/test_transport_channels.py) checks complete complex-amplitude dictionaries against the frozen evaluator for all three channels without filtering. Inverse evolution creates a large tail of roundoff-scale amplitudes. After each Hadamard, Ry or unitary-matrix primitive, the driver can remove amplitudes whose absolute magnitude is below the stated cutoff. The main run uses 1e-17 and the comparison uses 1e-16; neither rescales the surviving amplitudes. The state budget is 3,000,000 entries; absorption peaks at 2,654,208 entries. The step indices in each preparation checkpoint list run from 0 to 4. Each nonempty filtering event records the primitive index, the removed count and the Euclidean norm of the removed vector.

Under unitary propagation the sum ε of these event norms bounds the final vector perturbation due to filtering. For the retained vector of norm n, any projector probability changes by at most ε(2n+ε). This estimates the additional effect of filtering; ordinary floating-point gate arithmetic also contributes to the observed discrepancy. Each row requires the absolute discrepancy plus this bound to be strictly smaller than 2e-11. The same test applies to the normalization residual, the probability of any of the 17 work bits being nonzero, and the probability that the marker disagrees with the specified terminal status.

## Run from the project root in the QC environment

```bat
python -B tools\verify_transport_channels.py --out outputs/user_runs/channel_validation/result.json
python -B tools\verify_transport_channels.py --cutoff 1e-16 --out outputs/user_runs/channel_validation/cutoff_comparison.json
```

Use a new output filename for each run. The script exits successfully only when the requested channels complete and all checks pass. To reproduce the basis-budget exit, add `--cutoff 0 --max-states 300000 --channel detect` and use a separate output file.
