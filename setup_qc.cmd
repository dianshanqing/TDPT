@echo off
setlocal
cd /d "%~dp0"
REM Read-only by default; --install explicitly creates/aligns the quoted 'QC'.
if defined VRQO_CONDA_PYTHON goto :FOUND
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if exist "%%~dpI..\python.exe" set "VRQO_CONDA_PYTHON=%%~dpI..\python.exe"
if defined VRQO_CONDA_PYTHON goto :FOUND
if exist "%ProgramData%\anaconda3\python.exe" set "VRQO_CONDA_PYTHON=%ProgramData%\anaconda3\python.exe"
if defined VRQO_CONDA_PYTHON goto :FOUND
if exist "%USERPROFILE%\miniconda3\python.exe" set "VRQO_CONDA_PYTHON=%USERPROFILE%\miniconda3\python.exe"
:FOUND
if not defined VRQO_CONDA_PYTHON goto :MISSING
if not exist "%VRQO_CONDA_PYTHON%" goto :MISSING
if "%~1"=="" goto :CHECK
"%VRQO_CONDA_PYTHON%" -B "%~dp0qc_environment.py" %*
exit /b %ERRORLEVEL%
:CHECK
"%VRQO_CONDA_PYTHON%" -B "%~dp0qc_environment.py" --check
exit /b %ERRORLEVEL%
:MISSING
echo Conda BASE Python not found. Install Conda or set VRQO_CONDA_PYTHON to its python.exe.
exit /b 18
