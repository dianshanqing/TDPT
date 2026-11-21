# HIST-RRK-1E7

This four-qubit statevector sampling run uses three layers with transmission probabilities 0.1, a response target of 1e-07, and the recorded fixed Grover powers [0, 316, 790, 1581, 2372]. Across these powers, 10,000 observations cost 20,246,000 state-preparation-equivalent calls. The original JSON embeds the kernel, analytic target, MC tally and integer observations. `rrk_quantum_rows.csv` is the input for replaying the archived likelihood fit. Each `fit*` folder preserves its own fitted domain, estimate, interval and likelihood warnings.

The power schedule and restricted fitting domain were selected using the target. The response and query arithmetic describe calibration of this reduced kernel. Multiple parameter values can fit periodic amplified observations. Evidence for an unknown-probability, full three-dimensional transport estimation task would require a separate experiment.

The archive retains aggregate integer counts per power. Individual shot sequences and a hardware execution record are unavailable.
