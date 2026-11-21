@echo off
if not defined PYTHON set "PYTHON=python"
REM Stage 5 weighted rare-response MC cloud command template.
REM Edit HISTORIES/SEED or individual output labels if needed.
REM Weighted MC is the reference method; analog rare counts are diagnostic.

set "HISTORIES=1000000"
set "SEED=1234"

call %PYTHON% tools\rare_response_mc.py --steps 12 --histories %HISTORIES% --seed %SEED% --p-target 1e-6 --outdir outputs\extension\rare_response\mc_runs\s12_h%HISTORIES%_target1e-6_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 12 --histories %HISTORIES% --seed %SEED% --p-target 1e-8 --outdir outputs\extension\rare_response\mc_runs\s12_h%HISTORIES%_target1e-8_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 12 --histories %HISTORIES% --seed %SEED% --p-target 1e-10 --outdir outputs\extension\rare_response\mc_runs\s12_h%HISTORIES%_target1e-10_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 15 --histories %HISTORIES% --seed %SEED% --p-target 1e-6 --outdir outputs\extension\rare_response\mc_runs\s15_h%HISTORIES%_target1e-6_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 15 --histories %HISTORIES% --seed %SEED% --p-target 1e-8 --outdir outputs\extension\rare_response\mc_runs\s15_h%HISTORIES%_target1e-8_seed%SEED%
call %PYTHON% tools\rare_response_mc.py --steps 15 --histories %HISTORIES% --seed %SEED% --p-target 1e-10 --outdir outputs\extension\rare_response\mc_runs\s15_h%HISTORIES%_target1e-10_seed%SEED%
