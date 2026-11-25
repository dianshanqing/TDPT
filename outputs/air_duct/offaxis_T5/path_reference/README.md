# First-arrival path reference for the off-axis duct

From the project root, run the following standard-library calculation with a fresh output filename.

```console
python -B tools/check_offaxis_paths.py --config configs/models/q_duct_offaxis.json --out outputs/new_offaxis_reference/path_probabilities.json
```

The source at (1,1,1) and the detector at (3,2,2) differ by two cells along x and one each along y and z. A transport step moves a live particle by one axial cell. Four moves are therefore necessary, and a hit registers at the beginning of step five. The 12 distinct orderings of +x,+x,+y,+z exhaust the possible detecting trajectories at that horizon.

At the beginning of a move, a specified change of heading has weight `p_int * (1 - p_abs) / 6`. Keeping the same heading also includes the no-interaction branch of weight `1 - p_int`. The driver reads the cell material and single-group probabilities from the configuration and multiplies the four factors independently of the CSR recursion and quantum simulation. Each path lists its four cell coordinates, materials, headings, factors and any zero-factor explanation. Six sequences reach x=3 before completing the transverse moves and then require a turn in a zero-interaction cell, giving zero weight. The six remaining sequences all change direction at least three times, counting from the incident +x direction.

The sum of the six positive path weights is 0.00011062334505329146949. It is the unconditional five-step detection probability for this discrete, isotropically scattering model. The calculation tests both transverse displacements within a single history; it does not estimate an amplitude from sampled measurements.
