# Bond-dimension dependence of the 15-step baseline slab at optical thickness 23

The run uses the single-thread baseline configuration of the archived bond-256 case, with a maximum bond dimension of 64 and an explicit 1800 MiB limit for the supervised process. Both runs use truncation threshold 1e-16, double precision, the same transpiler basis, seed and optimization level. The original transport circuit, its compiled gate sequence and the compiled circuit with diagnostic saves have identical respective SHA-256 fingerprints across the two runs. The backend option dictionaries differ only in maximum bond dimension.

Run from the project root in the TDPT QC environment, with a fresh output directory.

```console
python -B main.py --phase mps_probe --suite outputs/deep_penetration/tau23/mps/baseline_tau23_T15_bond256_threads1_01/resolved_suite.json --mps-max-bond 64 --cpu-budget 1 --aer-threads 1 --mps-threads 1 --blas-threads 1 --run-memory-mib 1800 --outdir outputs/new_tau23_bond64
```

To repeat the numerical and identity checks on the distributed data, choose a fresh comparison filename.

```console
python -B tools/compare_tau23_bonds.py outputs/deep_penetration/tau23/mps/baseline_tau23_T15_bond64_mem1800/n8_w0002_tau23_T15_bond64_trunc1e-16.json.gz outputs/deep_penetration/tau23/mps/baseline_tau23_T15_bond256_threads1_01/n8_w0002_tau23_T15_bond256_trunc1e-16.json.gz --out outputs/new_comparison/bond_comparison.json
```

The per-setting gzip file contains the raw direct marginal probabilities, reference probabilities, work-bit diagnostics and Aer metadata. The comparison uses the independently archived T=15 classical row. No finite-count sampling was performed on the physical case.

| Quantity | Bond 64 | Bond 256 (archived) |
|---|---:|---:|
| Detection probability | 1.18471746213e-36 | 1.07289461381e-10 |
| Relative error in detection versus 1.07292089635e-10 | 100% | 0.002449624884% |
| Largest absolute error across the four status probabilities | 2.33259204618e-7 | 3.31854755481e-10 |
| Absolute normalization residual | 8.65973959208e-14 | 3.73144737331e-10 |
| Sum of nonzero work-bit marginals | 4.39254192417e-19 | 1.16446809078e-18 |
| Strict four-state check, tolerance 2e-11 | Failed | Failed |

At bond 64, the nearly normalized raw marginal distribution loses essentially the entire rare detector response. At bond 256, the detector agrees closely with its same-grid reference while the boundary and normalization residuals exceed the strict tolerance. These two settings quantify sensitivity for this particular circuit and grid.

Private run and library paths in the public JSON and decompressed per-setting JSON use repository-relative paths or <QC_ENV>. Storage hashes and sizes in the summary and comparison refer to the public bytes. Scientific values, circuit fingerprints and strict-check outcomes are preserved. The [source map](../../../../../provenance/source_files.csv) records both SHA-256 digests. The [initial attempt](../baseline_tau23_T15_bond64_initial/) stopped at the supervisor memory limit before a state distribution was available.
