# Reproduce Figure 1

[Figure 1](fig1_geometry.pdf) shows where particles start, where material is present and which lattice nodes count as a detection. Indices start at zero. Both sources point along +x.

| Panel | Model input | Grid and material | Detector |
| --- | --- | --- | --- |
| (a) Air duct | [legacy_duct.json](../../configs/models/legacy_duct.json) | 16 × 8 × 8. Source (1, 3, 3). Layers x = 8, 9 have a through-duct at y, z = 3, 4. | x = 12, four nodes at y, z = 3, 4. |
| (b) Uniform slab in vacuum | [Coarse slab presets](../../configs/models/compact_deep/) named `compact_n8_w0002_tau14`, `tau19` and `tau23` (same prefix) | 17 × 9 × 9. Source (2, 4, 4). Layers x = 4 through 11 span eight grid spacings. The three optical thicknesses share this geometry. | x = 14, nine nodes at y, z = 3, 4, 5. |

The input JSON files select model definitions from `presets.py`. The [drawing script](../../tools/plot_fig1_geometry.py) uses its own `AIR` and `UNIFORM` constants. The original drawing script defaults to the archived figures directory. From the project root, use the command below for a fresh run. It expands all four model inputs, checks the archived classical configurations against them, and compares the two drawings' source, material cells and detector bounds with the models. It then calls the drawing script, which draws the geometry directly. The transport probabilities are separate experimental results.

```bat
python -B tools\reproduce_figure1.py --outdir outputs/user_runs/geometry_001
```

Choose a new destination on each run. It receives separate panels and a combined figure, each as a PDF and a 600 dpi PNG. The command compares PNG pixels and PDF bytes (excluding creation time) with the archived exports in this directory. It also checks pixel dimensions, resolution and PDF page sizes. The render requires Matplotlib and Pillow. If they are missing from the selected environment, install them with `python -m pip install "matplotlib==3.10.6" "Pillow==12.0.0"`. The archived exports were reproduced with Matplotlib 3.10.6, FreeType 2.13.3 and Arial on Windows. Matplotlib 3.10.8 also produced the same page dimensions with some rendering differences. Font versions and the FreeType build can affect exact reproduction. Available fallbacks are Liberation Sans and DejaVu Sans. With matching rendering software, add `--require-exact` to return a nonzero exit code for any pixel or normalized PDF byte difference.

To reproduce the associated classical transport data, run these commands in Anaconda Prompt after the environment setup. The `reference` action calls `main.py`, which uses the float64 recurrence in [reference_solver.py](../../reference_solver.py) and writes `result.json`. Enter the loop directly in the prompt, using one percent sign.

```bat
call run.cmd reference --config configs/models/legacy_duct.json --steps 22 --outdir outputs/user_runs/fig1_duct
for %T in (14 19 23) do call run.cmd reference --config configs/models/compact_deep/compact_n8_w0002_tau%T.json --steps 15 --outdir outputs/user_runs/fig1_tau%T
```

Each `result.json` contains `config` and stepwise `rows`. Compare those fields with [the duct reference](../../outputs/air_duct/classical_reference/legacy_duct.json) or the matching `tau*/classical/compact_n8_w0002_tau*.json` under [the slabs](../../outputs/deep_penetration/README.md). The final `p_D` values are 0.4393621823 for the duct and 8.42142491 × 10⁻⁷, 5.75033655 × 10⁻⁹ and 1.07292090 × 10⁻¹⁰ for the three slabs in order. Generating the figure itself requires only the geometry.
