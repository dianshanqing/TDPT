"""Read-only Conda bootstrap, or explicitly requested locked installation.

Called with Conda BASE Python. python -m conda avoids dependency on the native
conda.exe launcher. No activation or renaming of the quoted QC is required.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from qc_versions import ENVIRONMENT_NAME as TARGET, PYTHON_VERSION as PYTHON, requirements
ROOT=Path(__file__).resolve().parent


def main(argv=None, *, runner=subprocess.run):
    ap=argparse.ArgumentParser(allow_abbrev=False)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--install',action='store_true')
    group.add_argument('--run',action='store_true')
    ap.add_argument('--outdir')
    ap.add_argument('arguments',nargs=argparse.REMAINDER)
    a=ap.parse_args(argv)
    if not a.run and a.arguments: ap.error('Unexpected setup arguments')
    conda=[sys.executable,'-m','conda']
    def execute(command, capture=False):
        return runner(command,cwd=ROOT,capture_output=capture,text=True,encoding='utf-8',errors='replace')
    def target_path():
        proc=execute([*conda,'env','list','--json'],True)
        if proc.returncode: raise RuntimeError('Conda env list failed: '+proc.stderr)
        matches=[Path(p) for p in json.loads(proc.stdout)['envs'] if Path(p).name==TARGET]
        if len(matches)>1: raise RuntimeError('Ambiguous quoted QC environments: '+str(matches))
        return matches[0] if matches else None
    try:
        version=execute([*conda,'--version'],True)
        if version.returncode:
            print('Conda unavailable in selected BASE Python. Install/repair Conda or set VRQO_CONDA_PYTHON.',file=sys.stderr)
            return version.returncode or 18
        print('Conda:',version.stdout.strip(),'| base Python:',sys.executable,flush=True)
        target=target_path()
        print('Target environment:',TARGET,'| path:',str(target) if target else 'MISSING',flush=True)
        if a.install:
            print('EXPLICIT INSTALL: Python '+PYTHON+' and exact versions from qc_versions.py',flush=True)
            print('\n'.join(requirements()),flush=True)
            command=([*conda,'create','-n',TARGET,'python='+PYTHON,'-y'] if target is None else
                     [*conda,'install','-p',str(target),'python='+PYTHON,'--freeze-installed','-y'])
            result=execute(command)
            if result.returncode: return result.returncode
            target=target_path()
            if target is None: raise RuntimeError('Conda reported success but target environment is missing')
            result=execute([str(target/'python.exe'),'-m','pip','install',*requirements()])
            if result.returncode: return result.returncode
        if target is None:
            print("Target 'QC' is missing. Run setup_qc.cmd --install explicitly; check/run never install.",file=sys.stderr)
            return 20
        python=target/'python.exe'
        if not python.is_file(): raise RuntimeError('Target Python missing: '+str(python))
        if a.run:
            rest=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
            return execute([str(python),'-B',str(ROOT/'run.py'),*rest]).returncode
        args=['check-env','--check']
        if a.outdir: args+=['--outdir',a.outdir]
        return execute([str(python),'-B',str(ROOT/'run.py'),*args]).returncode
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print('QC bootstrap failed: '+str(exc),file=sys.stderr)
        return 18


if __name__=='__main__': raise SystemExit(main())
