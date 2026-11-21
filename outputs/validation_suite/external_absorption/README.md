# Light pure-absorption comparison

This batch tests the transmission through two adjacent unit-thickness absorbers in continuous-flight Monte Carlo. A unit-weight neutron starts at (1, 2, 2) with direction +x. The two synthetic materials have macroscopic absorption cross sections 1 and 2 per unit length and zero scattering. Each particle can cross the detector plane at x = 5 at most once. The independent analytic transmission is exp(−1−2) = 0.04978706837. The corresponding TDPT lattice reference is the `REF-ABSORPTION_AB` entry.

The original batch input in `input/` contains the geometry, material densities, synthetic constant cross section, source, tally filters and seed. **Path-redacted copy:** in `input/input.xml`, the library and output directory fields use repository-relative locations in place of original private paths. Treat this file as the archived input and select a fresh output directory for any new calculation.

Open `raw/TallyResults.xml` for the native current tally and `result.json` for its interpretation. With 100 batches of 10,000 histories, the printed mean identifies 49,506 hits. The two-sided 95% Clopper–Pearson interval is [0.04908164591, 0.04993296589], which includes the analytic result. The original tally has an invalid processing instruction containing the string `?:anonymous`; `result.json` identifies the exact line ignored in memory during XML parsing.

`input_validation.json` and `verification.json` preserve the original input and tally checks. The distributed data consist of the experiment's input and measured output. The separate Light executable, its source tree and workstation launch records remain external. This comparison supports a straight, unscattered path with aligned material interfaces.
