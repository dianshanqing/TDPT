import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='nt','real Windows CMD integration')
class WindowsCmdTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cmd space ',dir=ROOT/'outputs')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)

    def cmd(self, filename, *args, env=None):
        command='cmd.exe /d /s /c ""'+str(ROOT/filename)+'" '+subprocess.list2cmdline(list(map(str,args)))+'"'
        return subprocess.run(command,cwd=self.folder,
            env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')

    def test_real_dry_override_and_nonempty_refusal(self):
        out=self.folder/'parsed output'
        args=['mps_probe','--dry-run','--counts-check','0','--seed','71','--cpu-budget','1','--mps-threads','1',
              '--run-memory-mib','512','--suite','baseline_tau19_probe','--mps-max-bond','128','--outdir',out]
        p=self.cmd('run.cmd',*args)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        r=next(json.loads(line)['request'] for line in p.stdout.splitlines() if line.startswith('{"status": "validated_only"'))
        self.assertEqual(r['task'],'mps_probe')
        self.assertEqual(r['backend']['bond_dimensions'],[128])
        self.assertEqual(r['cases'][0]['steps'],15)
        self.assertEqual(r['effective_suite']['seed'],71)
        self.assertFalse(r['effective_suite']['counts_check'])
        self.assertFalse(r['effective_suite']['sparse_crosscheck'])
        self.assertNotIn('histories',r['arguments'])
        self.assertTrue(r['dry_run'])
        self.assertFalse(out.exists())
        self.assertEqual(self.cmd('run.cmd',*args).returncode,0)
        self.assertFalse(out.exists())

    def test_real_default_50_shard_dry_run_and_occupied_refusal(self):
        out=self.folder/'new shard output'
        env=dict(os.environ,VRQO_CMD_REQUEST_JSON='{"shots": 999999}',
                 VRQO_SHARD_WORKER_TOKEN='stale',VRQO_DEADLINE_MONOTONIC='1')
        done=self.cmd('run.cmd','--outdir',out,env=env)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        self.assertIn('shards=50 parallel=50 shots=15000 chunk=300 seed=1234',done.stdout)
        self.assertIn('validated only, no scientific worker',done.stdout)
        self.assertNotIn('SUPERVISOR {',done.stdout)
        self.assertFalse(out.exists())
        out.mkdir();marker=out/'existing.txt';marker.write_text('retained')
        refused=self.cmd('run.cmd','--outdir',out,env=env)
        self.assertNotEqual(refused.returncode,0,refused.stdout+refused.stderr)
        self.assertIn('occupied',refused.stdout+refused.stderr)
        self.assertEqual(marker.read_text(),'retained')

    def test_real_unknown_and_inapplicable_fail(self):
        for args in [('reference','--bond-typo','1'),('reference','--mps-max-bond','16')]:
            p=self.cmd('run.cmd',*args)
            self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)

    def test_missing_conda_check_does_not_install(self):
        env=dict(os.environ,VRQO_CONDA_PYTHON=str(self.folder/'missing base python.exe'))
        p=self.cmd('setup_qc.cmd','--check',env=env)
        self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertIn('Conda',p.stdout+p.stderr)

    def test_controlled_missing_env_and_create_failure(self):
        # Controlled missing/install branch: no environment is installed or removed.
        import qc_environment
        calls=[]
        def fake(command,**kwargs):
            calls.append(command)
            if '--version' in command: return subprocess.CompletedProcess(command,0,'conda 26.1.1','')
            if 'list' in command: return subprocess.CompletedProcess(command,0,'{"envs": []}','')
            if 'create' in command: return subprocess.CompletedProcess(command,23,'','controlled failure')
            raise AssertionError(command)
        self.assertEqual(qc_environment.main(['--check'],runner=fake),20)
        self.assertFalse(any('create' in c for c in calls))
        self.assertEqual(qc_environment.main(['--install'],runner=fake),23)
        self.assertTrue(any(c[-5:]==['create','-n',"'QC'",'python=3.13.13','-y'] for c in calls))

    def test_controlled_alignment_pip_failure_is_propagated(self):
        import qc_environment
        import sys
        calls=[]
        def fake(command,**kwargs):
            calls.append(command)
            if '--version' in command: return subprocess.CompletedProcess(command,0,'conda test','')
            if 'list' in command: return subprocess.CompletedProcess(command,0,json.dumps({'envs':[sys.prefix]}),'')
            if 'pip' in command: return subprocess.CompletedProcess(command,31,'','controlled pip failure')
            return subprocess.CompletedProcess(command,0,'','')
        self.assertEqual(qc_environment.main(['--install'],runner=fake),31)
        self.assertTrue(any('python=3.13.13' in c and 'install' in c for c in calls))
        self.assertTrue(any('pip' in c and 'qiskit==2.5.2' in c for c in calls))
        calls.clear()
        self.assertEqual(qc_environment.main(['--check'],runner=fake),0)
        self.assertFalse(any('install' in c or 'create' in c for c in calls))


if __name__=='__main__': unittest.main()
