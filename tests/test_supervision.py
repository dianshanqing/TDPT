import json
import sys
import os
import time
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
from pathlib import Path


def temporary_outputs():
    return tempfile.TemporaryDirectory(dir=os.environ.get('VRQO_OUTPUT_DIR', Path(__file__).resolve().parents[1]/'outputs'))


class SupervisorTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows Job cleanup after process enumeration failure')
    def test_commit_exhaustion_in_discovery_still_terminates_owned_job(self):
        from supervised import run_limited
        import psutil
        # Reproduce the returned WinError 1455 without exhausting host memory.
        # Both monitoring and cleanup discovery fail; owned Job termination
        # must still run and the original execution failure must remain visible.
        error=OSError(22,'controlled commit exhaustion',None,1455)
        with temporary_outputs() as td:
            with patch.object(psutil.Process,'children',side_effect=error):
                r=run_limited([sys.executable,'-B','-c','import time; time.sleep(.5)'],
                    cwd=td,output_dir=Path(td)/'result',memory_bytes=256*1024**2)
            self.assertEqual(r['execution_status'],'supervisor_error',r)
            self.assertTrue(r['all_processes_reaped'],r)
            self.assertEqual(r['status'],'supervisor_error',r)
            self.assertTrue(r['cleanup']['job_confirmed_empty'],r)
            self.assertTrue(any(e['stage']=='discover_children' for e in r['cleanup']['events']),r)
            self.assertIn('1455',r['error'])

    @unittest.skipUnless(os.name=='nt','Windows suspended launch contract')
    def test_user_code_cannot_execute_before_job_assignment(self):
        from supervised import run_limited,_WindowsJob
        original=_WindowsJob.assign
        with temporary_outputs() as td:
            mark=Path(td)/'started'
            def checked_assign(job,proc):
                time.sleep(1)
                self.assertFalse(mark.exists(),'User code executed before memory/job limits were assigned')
                original(job,proc)
            with patch.object(_WindowsJob,'assign',checked_assign):
                r=run_limited([sys.executable,'-B','-c',"open('started','w').write('yes')"],cwd=td,output_dir=Path(td)/'result',timeout_sec=5,memory_bytes=256*1024**2)
            self.assertEqual(r['status'],'passed',r)
            self.assertTrue(mark.exists())

    def test_cleanup_failure_cannot_report_success(self):
        from supervised import run_limited
        with temporary_outputs() as td:
            with patch('supervised._kill_tree',side_effect=RuntimeError('cleanup not confirmed')):
                r=run_limited([sys.executable,'-B','-c','pass'],cwd=Path(td),output_dir=Path(td)/'result',timeout_sec=5,memory_bytes=256*1024**2)
            self.assertEqual(r['status'],'cleanup_failed')
            self.assertFalse(r['all_processes_reaped'])

    def test_timeout_reaps_spawned_child(self):
        from supervised import run_limited
        import psutil
        with temporary_outputs() as td:
            out=Path(td)
            code="import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); pathlib.Path('child.pid').write_text(str(p.pid)); time.sleep(60)"
            r=run_limited([sys.executable,'-B','-c',code],cwd=out,output_dir=out/'result',timeout_sec=3,memory_bytes=256*1024**2)
            self.assertEqual(r['status'],'timeout')
            self.assertFalse(psutil.pid_exists(int((out/'child.pid').read_text())))
            self.assertTrue(r['all_processes_reaped'])

    def test_nonzero_exit_is_failure(self):
        from supervised import run_limited
        with temporary_outputs() as td:
            r=run_limited([sys.executable,'-B','-c','raise SystemExit(7)'],cwd=Path(td),output_dir=Path(td)/'result',timeout_sec=5,memory_bytes=256*1024**2)
            self.assertEqual(r['status'],'failed')
            self.assertEqual(r['returncode'],7)

    def test_memory_limit_is_enforced(self):
        from supervised import run_limited
        with temporary_outputs() as td:
            code='a=[]\nwhile True: a.append(bytearray(8*1024**2))'
            r=run_limited([sys.executable,'-B','-c',code],cwd=Path(td),output_dir=Path(td)/'result',timeout_sec=10,memory_bytes=80*1024**2)
            self.assertNotEqual(r['status'],'passed')
            self.assertNotEqual(r['status'],'timeout')
            self.assertTrue(r['all_processes_reaped'])

    @unittest.skipUnless(os.name=='nt','Windows nested ownership Job')
    def test_ownership_only_job_retains_outer_memory_limit(self):
        from supervised import run_limited
        allocation='a=[]\nwhile True: a.append(bytearray(8*1024**2))'
        code=f"""
import json, subprocess, sys
from supervised import _WindowsJob
job=_WindowsJob(None)
try:
    print(json.dumps(job.memory_usage()),flush=True)
    child=subprocess.Popen([sys.executable,'-B','-c',{allocation!r}],creationflags=4)
    job.assign(child);job.resume(child)
    code=child.wait(timeout=10)
finally:job.close()
raise SystemExit(code)
"""
        with temporary_outputs() as td:
            r=run_limited([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[1],
                          output_dir=Path(td)/'result',timeout_sec=15,memory_bytes=96*1024**2)
            self.assertEqual(r['status'],'memory_limit',r)
            self.assertTrue(r['all_processes_reaped'],r)
            self.assertEqual(r['job_memory_limit_bytes'],96*1024**2)
            owner=json.loads((Path(td)/'result/stdout.log').read_text().strip())
            self.assertIsNone(owner['job_memory_limit_bytes'])


class OptionalDeadlineTests(unittest.TestCase):
    def run_short(self, td, **kwargs):
        from supervised import run_limited
        code="import os,json; print(json.dumps({'deadline':os.environ.get('VRQO_DEADLINE_MONOTONIC'),'threads':os.environ['OMP_NUM_THREADS']}))"
        try:
            return run_limited([sys.executable,'-B','-c',code],cwd=td,
                output_dir=Path(td)/'result',memory_bytes=256*1024**2,**kwargs)
        except (TypeError, ValueError) as exc:
            self.fail(f'Valid optional-deadline/resource request was rejected: {exc}')

    def test_no_timeout_has_null_metadata_and_no_fictitious_deadline(self):
        with temporary_outputs() as td, patch.dict(os.environ):
            os.environ.pop('VRQO_DEADLINE_MONOTONIC',None)
            r=self.run_short(td,timeout_sec=None,threads=4)
            self.assertEqual(r['status'],'passed',r)
            self.assertIsNone(r['timeout_seconds'])
            self.assertIsNone(r['effective_deadline_monotonic'])
            child=json.loads((Path(td)/'result/stdout.log').read_text())
            self.assertEqual(child,{'deadline':None,'threads':'4'})
            self.assertIsNone(json.loads((Path(td)/'result/process_result.json').read_text())['timeout_seconds'])

    def test_omitted_timeout_has_no_wall_clock_deadline(self):
        import inspect
        from supervised import run_limited
        self.assertIsNone(inspect.signature(run_limited).parameters['timeout_sec'].default)

    def test_no_timeout_preserves_inherited_parent_deadline(self):
        with temporary_outputs() as td, patch.dict(os.environ):
            deadline=time.monotonic()+10
            os.environ['VRQO_DEADLINE_MONOTONIC']=str(deadline)
            r=self.run_short(td,timeout_sec=None)
            self.assertEqual(r['status'],'passed',r)
            self.assertEqual(r['effective_deadline_monotonic'],deadline)
            self.assertEqual(float(json.loads((Path(td)/'result/stdout.log').read_text())['deadline']),deadline)

    def test_expired_parent_is_not_extended_or_started(self):
        with temporary_outputs() as td, patch.dict(os.environ):
            os.environ['VRQO_DEADLINE_MONOTONIC']=str(time.monotonic()-1)
            r=self.run_short(td,timeout_sec=10)
            self.assertEqual(r['status'],'timeout',r)
            self.assertIsNone(r['returncode'])
            self.assertEqual((Path(td)/'result/stdout.log').read_text(),'')

    def test_keyboard_interrupt_records_and_reaps_actual_child(self):
        from supervised import run_limited
        import psutil
        real_sleep=time.sleep
        with temporary_outputs() as td, patch.dict(os.environ):
            os.environ.pop('VRQO_DEADLINE_MONOTONIC',None)
            marker=Path(td)/'child.pid'
            deadline=time.monotonic()+5
            interrupted_once=False
            def interrupt_when_child_exists(seconds):
                nonlocal interrupted_once
                if marker.exists() and not interrupted_once:
                    interrupted_once=True
                    raise KeyboardInterrupt('controlled Ctrl+C')
                if time.monotonic()>deadline:raise AssertionError('Child did not start')
                real_sleep(seconds)
            code="import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']); pathlib.Path('child.pid').write_text(str(p.pid)); time.sleep(30)"
            try:
                with patch('supervised.time.sleep',side_effect=interrupt_when_child_exists):
                    r=run_limited([sys.executable,'-B','-c',code],cwd=td,output_dir=Path(td)/'result',timeout_sec=10,memory_bytes=256*1024**2)
            except KeyboardInterrupt:
                self.fail('Ctrl+C escaped without an interrupted process report')
            self.assertEqual(r['status'],'interrupted',r)
            self.assertTrue(r['interrupted'])
            self.assertTrue(r['all_processes_reaped'],r)
            self.assertFalse(psutil.pid_exists(int(marker.read_text())))
            self.assertIn('KeyboardInterrupt',(Path(td)/'result/stderr.log').read_text())

    def test_optional_progress_echo_preserves_both_logs(self):
        from supervised import run_limited
        with temporary_outputs() as td:
            stdout=io.StringIO();stderr=io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    r=run_limited([sys.executable,'-B','-c',"import sys;print('START short',flush=True);print('diagnostic',file=sys.stderr,flush=True)"],cwd=td,output_dir=Path(td)/'result',timeout_sec=5,memory_bytes=256*1024**2,echo_output=True)
            except TypeError as exc:self.fail(f'Progress echo request was rejected: {exc}')
            self.assertEqual(r['status'],'passed',r)
            self.assertIn('START short',stdout.getvalue())
            self.assertIn('diagnostic',stderr.getvalue())
            self.assertIn('START short',(Path(td)/'result/stdout.log').read_text())

    def test_invalid_nonfinite_limits_and_noninteger_threads_are_rejected(self):
        from supervised import run_limited
        for kwargs in ({'timeout_sec':float('nan')},{'timeout_sec':float('inf')},{'threads':1.5},{'threads':0}):
            with self.subTest(kwargs=kwargs), temporary_outputs() as td:
                with self.assertRaises(ValueError):
                    run_limited([sys.executable,'-B','-c','pass'],cwd=td,output_dir=Path(td)/'result',memory_bytes=256*1024**2,**kwargs)

    def test_output_protection_still_rejects_old_and_existing_directories(self):
        from supervised import run_limited
        with temporary_outputs() as td:
            existing=Path(td)/'existing';existing.mkdir();(existing/'sentinel').write_text('keep')
            for out in (existing,Path(td)/'pre-outputs'/'new'):
                with self.assertRaises((ValueError,FileExistsError)):
                    run_limited([sys.executable,'-B','-c','pass'],cwd=td,output_dir=out,timeout_sec=1)
            self.assertEqual((existing/'sentinel').read_text(),'keep')

@unittest.skipUnless(os.name=='nt','Actual Windows Job Object cleanup')
class WindowsRealProcessCleanupTests(unittest.TestCase):
    """Small real process trees; no mocked process discovery or termination."""

    _record_self = (
        "import json, pathlib, psutil\n"
        "self_process = psutil.Process()\n"
        "pathlib.Path(MARKER).write_text(json.dumps({"
        "'pid': self_process.pid, 'create_time': self_process.create_time()}))\n"
    )

    def _run_scripts(self, scripts, markers, timeout_sec=5):
        from supervised import run_limited
        import psutil
        with temporary_outputs() as td:
            directory=Path(td)
            for name,code in scripts.items():
                (directory/name).write_text(code,encoding='utf8')
            result=run_limited([sys.executable,'-B','parent.py'],cwd=directory,
                output_dir=directory/'result',timeout_sec=timeout_sec,
                memory_bytes=256*1024**2,threads=1)
            observations={}
            for name in markers:
                path=directory/name
                if not path.exists():
                    observations[name]={'marker_missing':True}
                    continue
                identity=json.loads(path.read_text())
                pid=identity['pid']
                exists=psutil.pid_exists(pid)
                same_identity=False
                if exists:
                    try:
                        same_identity=psutil.Process(pid).create_time()==identity['create_time']
                    except psutil.NoSuchProcess:
                        exists=False
                observations[name]={**identity,'pid_exists_after_run':exists,
                    'same_identity_after_run':same_identity}
            details_path=directory/'details.json'
            details=json.loads(details_path.read_text()) if details_path.exists() else None
            evidence={'test':self.id(),'result':result,'pid_observations':observations,
                'details':details,'scripts':scripts,
                'stdout':(directory/'result/stdout.log').read_text(encoding='utf8'),
                'stderr':(directory/'result/stderr.log').read_text(encoding='utf8')}
            if os.environ.get('VRQO_OUTPUT_DIR'):
                evidence_path=Path(os.environ['VRQO_OUTPUT_DIR'])/(self._testMethodName+'.json')
                with evidence_path.open('x',encoding='utf8') as stream:
                    json.dump(evidence,stream,ensure_ascii=False,indent=2,allow_nan=False)
            # Preserve the measured failure before best-effort emergency cleanup.
            # A reused PID must never be killed by this test fixture.
            for observed in observations.values():
                if observed.get('same_identity_after_run'):
                    try:
                        process=psutil.Process(observed['pid'])
                        if process.create_time()==observed['create_time']:
                            process.kill();process.wait(timeout=5)
                    except psutil.NoSuchProcess:
                        pass
            for name,observed in observations.items():
                self.assertNotIn('marker_missing',observed,(name,evidence))
                self.assertFalse(observed['pid_exists_after_run'],(name,evidence))
            self.assertTrue(result['all_processes_reaped'],evidence)
            return result,details

    def test_natural_exit_accepts_already_finished_historical_child(self):
        # A descendant disappearing before parent exit is successful cleanup,
        # and must not turn a normal zero exit into a cleanup failure.
        child="MARKER='child.json'\n"+self._record_self+"import time\ntime.sleep(.20)\n"
        parent="MARKER='parent.json'\n"+self._record_self+(
            "import subprocess, sys, time\n"
            "child=subprocess.Popen([sys.executable,'-B','child.py'])\n"
            "exit_code=child.wait(timeout=3)\n"
            "pathlib.Path('details.json').write_text(json.dumps({'child_returncode':exit_code}))\n"
            "time.sleep(.15)\n"
        )
        result,details=self._run_scripts({'parent.py':parent,'child.py':child},
            ['parent.json','child.json'])
        self.assertEqual(result['status'],'passed',result)
        self.assertEqual(result['returncode'],0,result)
        self.assertEqual(details,{'child_returncode':0})

    def _nested_tree(self, timeout):
        grandchild="MARKER='grandchild.json'\n"+self._record_self+"import time\ntime.sleep(30)\n"
        child="MARKER='child.json'\n"+self._record_self+(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable,'-B','grandchild.py'])\n"
            "time.sleep(30)\n"
        )
        parent="MARKER='parent.json'\n"+self._record_self+(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable,'-B','child.py'])\n"
            "ready_deadline=time.monotonic()+3\n"
            "while not pathlib.Path('grandchild.json').exists():\n"
            "    if time.monotonic()>ready_deadline: raise RuntimeError('Grandchild did not start')\n"
            "    time.sleep(.01)\n"
            "time.sleep(.15)\n"
        )+("time.sleep(30)\n" if timeout else '')
        return self._run_scripts({'parent.py':parent,'child.py':child,'grandchild.py':grandchild},
            ['parent.json','child.json','grandchild.json'],timeout_sec=3 if timeout else 5)[0]

    def test_normal_parent_exit_reaps_child_and_grandchild(self):
        # Parent completion alone cannot establish that the three-level tree ended.
        result=self._nested_tree(timeout=False)
        self.assertEqual(result['status'],'passed',result)
        self.assertEqual(result['returncode'],0,result)

    def test_timeout_reaps_child_and_grandchild(self):
        result=self._nested_tree(timeout=True)
        self.assertEqual(result['status'],'timeout',result)
        self.assertIsNotNone(result['returncode'],result)
        self.assertNotEqual(result['returncode'],0,result)

    def test_repeated_short_child_exit_races_do_not_report_cleanup_failure(self):
        # Different short lifetimes cross the supervisor polling interval and
        # leave historical descendants that may vanish during inspection.
        child=("import sys\nMARKER='child_'+sys.argv[1]+'.json'\n"+self._record_self+
            "import time\ntime.sleep(float(sys.argv[2]))\n")
        parent="MARKER='parent.json'\n"+self._record_self+(
            "import subprocess, sys, time\n"
            "returncodes=[]\n"
            "for index,delay in enumerate([0,.01,.035,.08,0,.015,.05,.10]):\n"
            "    child=subprocess.Popen([sys.executable,'-B','child.py',str(index),str(delay)])\n"
            "    returncodes.append(child.wait(timeout=3))\n"
            "pathlib.Path('details.json').write_text(json.dumps({'child_returncodes':returncodes}))\n"
            "time.sleep(.10)\n"
        )
        result,details=self._run_scripts({'parent.py':parent,'child.py':child},
            ['parent.json']+[f'child_{index}.json' for index in range(8)],timeout_sec=8)
        self.assertEqual(result['status'],'passed',result)
        self.assertEqual(result['returncode'],0,result)
        self.assertEqual(details,{'child_returncodes':[0]*8})


if __name__=='__main__': unittest.main()
