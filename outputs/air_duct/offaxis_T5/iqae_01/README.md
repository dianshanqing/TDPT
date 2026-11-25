# Budget-limited adaptive amplitude estimation of off-axis detection

The sampler evolves the marked five-step baseline circuit and the published Grover operator gate by gate in the sparse basis. It draws fresh ideal binomial shots from the flag probability at each admitted power. The circuit identity in [protocol.json](protocol.json) matches the separate [five-step gate experiment](../gate_02/identity.json). The classical probability is used only in subsequent verification.

The [input protocol](../../../../configs/offaxis_iqae_budget.json) fixes the seed, batch size, risk spending, stopping condition and resource budgets before sampling. The pilot value was 15.4744 seconds per Grover iterate. The 900-second limit applies to admission of a new power. Admission uses the largest of the pilot time and observed per-iterate times, multiplied by 1.25. A running iterate can finish later. Timing and available memory can change the last admitted power on other machines.

| Power | Flag-one outcomes | Shots | Direct sparse flag probability |
|---:|---:|---:|---:|
| 0 | 0 | 512 | 0.00011062334505329156 |
| 7 | 14 | 512 | 0.024685331246565204 |

The adaptive controller next selected power 50. After 99.23 seconds, it estimated that extending the prefix from 7 to 50 would take another 831.75 seconds with the declared safety factor. It stopped before executing that extension. The last valid interval for the unamplified probability at the nominal 95% confidence level under ideal sampling was [0.0000535064850270209, 0.0002371707330607917]. Its relative stopping statistic was 1.71628 against a target of 0.01. The 1,024 shots correspond to 4,608 marked preparations and 3,584 inverses, or 8,192 calls in total. The simulator evolved one preparation and seven successive Grover iterates and reused the prefix. The query count describes the sampled circuits.

[Result and intervals](result.json) contains every round, its risk allocation, the admitted-power checks and the exit condition. The two batch files record the actual draws. The seven Q files record the sparse sizes, timings and each amplitude-screening event. At power 7 the norm residual, nonzero work probability and mark mismatch were all zero, and the cumulative screening bound for a projector probability was 4.37 × 10⁻¹⁵.

Run from the TDPT root in the QC environment using a fresh destination. The pilot time below reproduces this run's admission input.

```bat
python -B tools\run_offaxis_iqae.py --pilot-seconds 15.4744 --out outputs/user_runs/offaxis_iqae_new
```
