@echo off
setlocal
REM ==== FIXED PARAMETER FORM. Edit values; keep the launcher below unchanged. ====
REM [1] Mode and model. Select CONFIG or SUITE, never both. Blank inherits defaults.
REM TASK: test / reference / mc / mps_probe / iqae / mps_batch / validate / circuit / summarize
set "TASK=test"
set "CONFIG=legacy_duct"
set "SUITE="
set "CASE_ID="
set "STEPS=15"
set "OUTDIR="
set "DRY_RUN=1"
REM Blank OUTDIR chooses a fresh unique directory; occupied output is never overwritten.
REM [1b] test/mc shards. Sequential process creation still permits full concurrency.
set "SHARD_MODE=1"
set "NUM_SHARDS=50"
set "MAX_PARALLEL=50"
set "SHOT_CHUNK=300"
set "TEST_MEASURE_X=1"
set "TEST_MEASURE_DIR=1"
set "TEST_MEASURE_WORK=0"
set "AER_SAFE_BASIS=1"
REM One batch heartbeat; resource sampling and process polling are separate clocks.
set "AER_HEARTBEAT_SEC=60"
set "RESOURCE_SAMPLE_SEC=1"
set "POLL_SEC=0.1"
set "DEBUG_RUNTIME=0"
REM [2] Shards share one aggregate batch memory guard; CPU_BUDGET is per science worker.
set "CPU_BUDGET=1"
set "MEMORY_MIB=98304"
set "SEED=1234"
REM [3] Quantum circuit and numerical controls. These are not statistical accuracy.
set "PROFILE=baseline"
set "BOND=896"
set "MPS_TRUNC=1e-16"
set "AER_THREADS=1"
set "MPS_THREADS=1"
set "BLAS_THREADS=1"
set "MPS_LAPACK=0"
set "COUNTS_CHECK=0"
REM [4] Sampling. Blank SHOTS uses the mode/suite default: IQAE=512 per round.
REM HISTORIES applies only to mc; blank uses 200000. reference has no sampling.
set "SHOTS=15000"
set "HISTORIES="
REM [5] IQAE only. Relative error/confidence cover one complete estimation.
set "RELATIVE_ERROR=0.01"
set "CONFIDENCE=0.95"
set "IQAE_MAX_SHOTS=20000"
set "IQAE_MAX_QUERIES=10000000"
set "IQAE_MAX_GROVER_POWER=256"
REM Optional fixed-power diagnostic: blank runs the estimator; compile / probability / sparse.
REM compile exports circuits without sampling. probability reads target diagnostics only.
set "IQAE_DIAGNOSTIC="
set "IQAE_POWERS="
REM Blank diagnostic budgets use 60000 sparse states / 4000000 compiled gates.
set "IQAE_MAX_SPARSE_STATES="
set "IQAE_MAX_COMPILED_GATES="
REM Existing diagnostic export directory for exact local/cloud gate comparison.
set "IQAE_COMPARE_TO="
REM [6] Optional modes. Ignored by ordinary single-task calculations.
REM mps_batch only: concurrent task caps plus 512 MiB coordinator reserve.
set "BATCH_WORKERS=2"
set "BATCH_MEMORY_MIB=16896"
REM summarize only. Leave OUTDIR blank to write inside INPUT/summary.
set "INPUT="
REM Optional fixed-prefix performance diagnostic. Blank preserves full T15.
set "PERF_PREFIX_GATES="
set "PERF_REFERENCE="
REM ==== END PARAMETERS. CLI overrides edited values; no automatic install. ====
set "VRQO_RUN_CMD=1"
REM A public launch starts a new invocation, not a stale internal worker.
set "VRQO_CMD_REQUEST_JSON="
set "VRQO_MAIN_WORKER_TOKEN="
set "VRQO_WORKER_STARTED_FILE="
set "VRQO_OUTPUT_DIR="
set "VRQO_SHARD_WORKER_TOKEN="
set "VRQO_SUPERVISED="
set "VRQO_DEADLINE_MONOTONIC="
set "VRQO_PARENT_STATUS_FILE="
set "VRQO_PROGRESS_FILE="
set "VRQO_RESOURCE_FILE="
cd /d "%~dp0"
if defined VRQO_CONDA_PYTHON goto :FOUND
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if exist "%%~dpI..\python.exe" set "VRQO_CONDA_PYTHON=%%~dpI..\python.exe"
if defined VRQO_CONDA_PYTHON goto :FOUND
if exist "%ProgramData%\anaconda3\python.exe" set "VRQO_CONDA_PYTHON=%ProgramData%\anaconda3\python.exe"
if defined VRQO_CONDA_PYTHON goto :FOUND
if exist "%USERPROFILE%\miniconda3\python.exe" set "VRQO_CONDA_PYTHON=%USERPROFILE%\miniconda3\python.exe"
:FOUND
if not defined VRQO_CONDA_PYTHON goto :MISSING
if not exist "%VRQO_CONDA_PYTHON%" goto :MISSING
REM Parse launch and return together, so later edits cannot shift the resume offset.
REM CALL expands ERRORLEVEL after Python returns, rather than when the block is read.
(
"%VRQO_CONDA_PYTHON%" -B "%~dp0qc_environment.py" --run -- %*
call exit /b %%ERRORLEVEL%%
)
:MISSING
echo Conda BASE Python not found. Open Anaconda Prompt or set VRQO_CONDA_PYTHON to its python.exe.
exit /b 18
