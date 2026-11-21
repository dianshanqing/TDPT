@echo off
if not defined PYTHON set "PYTHON=python"
REM Weighted rare-response MC templates for the newly added 1e-7 and 1e-9 targets only.
REM Existing 1e-6, 1e-8, and 1e-10 runs are intentionally not repeated here.

set "HISTORIES=1000000"
set "SEED=1234"

call %PYTHON% tools\rare_response_mc.py --steps 12 --histories %HISTORIES% --seed %SEED% --p-target 1e-7 --outdir outputs\extension\rare_response\mc_runs\s12_h%HISTORIES%_target1e-7_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 12 --histories %HISTORIES% --seed %SEED% --p-target 1e-9 --outdir outputs\extension\rare_response\mc_runs\s12_h%HISTORIES%_target1e-9_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 15 --histories %HISTORIES% --seed %SEED% --p-target 1e-7 --outdir outputs\extension\rare_response\mc_runs\s15_h%HISTORIES%_target1e-7_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 15 --histories %HISTORIES% --seed %SEED% --p-target 1e-9 --outdir outputs\extension\rare_response\mc_runs\s15_h%HISTORIES%_target1e-9_seed%SEED%
