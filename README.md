# Three-Dimensional Particle Transport (TDPT)

TDPT simulates finite-step particle transport on a three-dimensional Cartesian lattice. Use it to calculate detection, absorption, leakage and survival probabilities, compare a reversible quantum circuit with independent classical recursion, and measure the circuit resources required by different history encodings. The distribution includes an air duct through a slab, homogeneous and layered slabs, and small geometries with known limiting behaviour.

## Physical model

A history carries position, one of six axial directions, an energy-group label and a status. At the start of a step, a live particle on the outer boundary leaks. The remaining live particles are tested for detection on a counting plane with inclusive transverse bounds. The material in the current cell sets the probabilities of interaction, conditional absorption and scatter. Scattering redistributes the direction over the six axes. Surviving particles move one lattice spacing. A particle reaching the counting plane on the last move can first register on the next step. Terminated histories retain their physical state.

The observables refer to this discrete model and observation horizon. The named material slots in the examples carry numerical interaction parameters. The deep-penetration benchmarks use a fixed energy group. The duct regression retains two energy labels with zero downscatter. The circuit supports fixed-group evolution within its declared capabilities. For grid sensitivity, compare classical runs at matched physical dimensions and observation lengths. The remaining survival mass bounds the possible increase in detection with additional steps.

The [geometry diagram](figures/fig1/fig1_geometry.pdf) shows the full-size air duct and the uniform slab shared by the three deep-penetration benchmarks. The [geometry guide](figures/fig1/README.md) connects the two panels to their model inputs and classical results, and gives the command to redraw and compare the six archived files.

## Run on Windows

Use Anaconda Prompt **(cmd.exe)**. Extract the complete project into a writable location. In the example below, change `C:\work\TDPT` to the directory containing `run.cmd`. Run the commands one line at a time.

```bat
conda activate base
set "VRQO_CONDA_PYTHON=%CONDA_PREFIX%\python.exe"
cd /d "C:\work\TDPT"
call setup_qc.cmd --check
echo %ERRORLEVEL%
```

The check reports the target interpreter, installed package versions and import results. TDPT's transport runtime uses Python 3.13.13 and six pinned packages (Qiskit, Qiskit Aer, Qiskit Algorithms, NumPy, SciPy and psutil). The [version lock](qc_versions.py) lists their exact versions. If the check reports a missing environment or mismatched dependencies, run the explicit setup command from the same prompt. Setup requires access to the configured Conda and pip package sources and permission to create or update the target environment.

```bat
call setup_qc.cmd --install
echo %ERRORLEVEL%
```

Run the check again and confirm a return code of **0** before continuing. The environment name in the frozen launcher includes two literal apostrophes. Enter the activation line exactly as shown. Keep `VRQO_CONDA_PYTHON` pointing at the **base** interpreter after activation.

```bat
call setup_qc.cmd --check
echo %ERRORLEVEL%
conda activate "'QC'"
python -B tools\verify_release.py
call run.cmd
echo %ERRORLEVEL%
```

The supplied `run.cmd` sets `DRY_RUN=1`. Its default check resolves the full-size duct at 15 steps, 50 shards, 15,000 total shots and seed 1234. Look for `validated only, no scientific worker` and exit code 0. Values in the parameter block are assigned inside the CMD file. To run the selected calculation, edit **a working copy of the project** and change its `set "DRY_RUN=1"` line to `set "DRY_RUN=0"` after reviewing the dry run. The default 15-step sampled circuit can be expensive.

For plotting and developer tests, install the additional packages in the activated QC environment when needed. Matplotlib 3.10.6 is the renderer used for the distributed figure. The figure guide describes font requirements for reproducing the original pixels. Pytest 9.0.3 includes support for the subtests used here.

```bat
python -m pip install matplotlib==3.10.6 Pillow==12.0.0 pytest==9.0.3
python -m pip check
```

### A small calculation

From the same prompt, run a three-step classical reference. An explicit action different from the one selected in the CMD form uses that action's own defaults. The following sequence previews and then executes the small case.

```bat
call run.cmd reference --config q_uniform_small --steps 3 --dry-run --outdir outputs/user_runs/small_reference
call run.cmd reference --config q_uniform_small --steps 3 --outdir outputs/user_runs/small_reference
echo %ERRORLEVEL%
python -c "import json; r=json.load(open('outputs/user_runs/small_reference/result.json')); print(r['rows'][-1]['probabilities'])"
```

The last array gives survival, detection, absorption and boundary leakage, in that order. The expected detection probability is 0.5023729262121445. Earlier rows give all four channels at each step. `resolved_config.json` gives the geometry and material parameters. To repeat, choose a fresh final directory name. Use the channel labels in each output when reading counts or distributions.

To build a custom input, export an expanded model from the project root in the same prompt. Edit the new JSON, then pass its path with `--config`. A preset selector contains only the `preset` key. The expanded JSON accepts `geom` (grid dimensions, inclusive slab and detector indices), `source` (cell, direction and group), `groups`, `mats`, `model`, `layers` and grid spacing `h`. Each material supplies one entry per energy group for interaction probability `p_int`, conditional absorption `p_abs` and downscatter probability `q_down`. The loader validates dimensions, bounds and probabilities. The step count in the examples is selected on the command line.

```bat
python -B -c "import json; from models import load_document; f=open('my_slab.json','x'); json.dump(load_document('q_uniform_small'),f,indent=2); f.close()"
call run.cmd reference --config my_slab.json --steps 3 --dry-run --outdir outputs/user_runs/my_slab
```

### Select another calculation

| Setting in `run.cmd` | What it controls |
| --- | --- |
| `TASK`, `CONFIG`, `SUITE`, `CASE_ID`, `STEPS` | Choose an action and a model preset or a suite; select one input form at a time. Preset JSON files in `configs/` resolve through `models.py` and `presets.py`. |
| `PROFILE`, `BOND`, `MPS_TRUNC` | History encoding, maximum MPS bond dimension and truncation threshold. These affect circuit size or numerical accuracy. |
| `SHOTS`, `HISTORIES`, `SEED` | Measurements, classical Monte Carlo histories and random seed as applicable. |
| `SHARD_MODE`, `NUM_SHARDS`, `MAX_PARALLEL`, thread and memory settings | Local parallelism and simulation budgets. Start with a small case on a workstation. |
| `RELATIVE_ERROR`, `CONFIDENCE`, IQAE budgets | Interval stopping target and limits on measurements, state-preparation calls and Grover power. |
| `OUTDIR` | A blank value generates a fresh path under `outputs/`. For an explicit destination, use a new folder under `outputs/user_runs/`. The entry point rejects an occupied run directory. |

Use `call run.cmd reference --help` or the corresponding action's help for CLI options. The [benchmark guide](outputs/README.md) gives commands for classical recursion, sampled transport, gate validation, MPS readout, compilation and IQAE replay, with the files to inspect for each. `shard_runner.py` and `merge_shards.py` implement sharded runs and consistency checks.

Run the distributed tests from a disposable project copy using its full Windows directory name. The gate trace matches the source filename to its resolved path. A DOS 8.3 alias can suppress step-boundary observations. Use a fresh pytest directory under `outputs/user_runs`. A short check and the complete collection are shown below.

```bat
if not exist outputs\user_runs mkdir outputs\user_runs
python -B -m pytest -q -p no:cacheprovider --basetemp=outputs/user_runs/pytest_short tests/test_compact_reference.py tests/test_relative_iqae.py tests/test_iqae_resources.py
python -B -m pytest -q -p no:cacheprovider --basetemp=outputs/user_runs/pytest_all tests
```

The frozen developer runner `run_local_validation.py` also accepts named groups. Its original default list includes developer modules outside this distribution. To run a distributed group with the original isolation and reporting, use `python -B run_local_validation.py --group tests.test_root_runtime --outdir outputs/user_runs/dev_group`. Two historical collectors in `tests/collect_*_evidence.py` document how earlier batches were produced. Their audit and catalog stages use the historical output tree.

## Read and verify the archived experiments

Start with [the data guide](outputs/README.md). Look up a model, method or evidence ID in [the evidence index](outputs/evidence_index.csv). Paths in the index start at this repository root. Consult the original per-case record for the resolved input, precision settings, observable definition and outcome. A direct MPS probability, a sampled terminal count and an IQAE confidence interval describe different measurements.

Run `python -B tools/verify_release.py` on a fresh extraction to verify the SHA-256 manifest and resolve the evidence links. The [file guide](provenance/file_roles.csv) identifies the purpose of each distributed file. [The freeze record](provenance/runtime_freeze.json) identifies the 44 root Python files and the frozen launcher. Run identities hash every root Python module, including diagnostic and compatibility entry points. Paths in the [source map](provenance/source_files.csv) describe the original collection. For current input and result locations, use the evidence index. The latest short-range sparse IQAE data were produced after the runtime freeze. The [original execution driver](iqae_sparse_complete.py) matches its recorded SHA-256, and all 45 entries in the successful batch's production-source manifest match this project. The [IQAE guide](outputs/amplitude_estimation/README.md) gives separate commands to preview a fresh execution and replay archived counts. Version 2.0.1 adds local verification records alongside the dated calculations. Each record identifies the code and numerical settings used for that calculation.

## Citation and license

Software attribution and version are in [CITATION.cff](CITATION.cff). The software is distributed under the [MIT License](LICENSE). The setup script installs dependencies in Conda; their respective package distributions carry their own license notices.

## Run the additional verification cases

Use a fresh extraction and the activated QC interpreter. On a clean checkout or extraction, `--strict-tree` checks for unlisted files as well as missing or modified files. Run that check before creating your own results.

```bat
python -B tools/verify_release.py --strict-tree
python -B tools/verify_transport_channels.py --out outputs/user_runs/duct_channels/result.json
python -B tools/verify_short_oracle.py --out-dir outputs/user_runs/short_oracle
python -B outputs/deep_penetration/tau23/path_classes/tagged_tau23.py --check
```

Choose fresh destinations for repeated runs. See [reduced duct channel marking](outputs/air_duct/channel_marking/baseline_T5/README.md) for the six flag probabilities and the numerical filters, [two-cell detector oracle](outputs/amplitude_estimation/tau19_two_cell/oracle_audit_v2.0.1/README.md) for the local compilation and protocol checks, and [short transport amplitude estimation](outputs/amplitude_estimation/README.md) for the original execution and count-replay commands.
