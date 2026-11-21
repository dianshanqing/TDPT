"""Deterministic OS-boundary races; actual Windows trees live in test_supervision."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import psutil
import supervised

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='nt','Windows Job-owned cleanup contract')
class CleanupIdentityTests(unittest.TestCase):
    def cleanup_case(self,states=('live',),created=10.):
        # Inject only the OS identity boundary. Never kill/wait this fake PID.
        pid=987654321;sequence=list(states);state={'value':sequence[0]};actions=[]
        old=Mock(spec=psutil.Process);old.pid=pid;old.create_time.return_value=10.
        old.kill.side_effect=lambda:actions.append('old_kill')
        old.wait.side_effect=lambda **kw:actions.append('old_wait')
        real_process=psutil.Process;real_exists=psutil.pid_exists
        def exists(number):
            return state['value']!='gone' if number==pid else real_exists(number)
        def snapshot(number):
            if number!=pid:return real_process(number)
            state['value']=sequence.pop(0) if len(sequence)>1 else sequence[0]
            if state['value']=='gone':raise psutil.NoSuchProcess(pid)
            if state['value']=='denied':raise psutil.AccessDenied(pid)
            if state['value']=='stale_nsp':raise psutil.NoSuchProcess(pid)
            fresh=Mock(spec=real_process);fresh.pid=pid
            fresh.create_time.return_value=20. if state['value']=='reused' else 10.
            fresh.kill.side_effect=lambda:actions.append('fresh_kill')
            fresh.wait.side_effect=lambda **kw:actions.append('fresh_wait')
            return fresh
        with subprocess.Popen([sys.executable,'-B','-c','pass'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) as parent:
            parent.wait(timeout=10)
            with patch.object(supervised.psutil,'Process',side_effect=snapshot),patch.object(supervised.psutil,'pid_exists',side_effect=exists):
                try:report=supervised._kill_tree(parent,{(pid,created):old},wait_timeout=.02)
                except Exception as exc:self.fail(f'Per-process cleanup must classify this race instead of aborting: {exc!r}')
        self.assertIsInstance(report,dict,'Cleanup must return explicit confirmed/unconfirmed outcomes')
        self.assertEqual(actions,[],'Windows historical PIDs must not be killed or waited through newly opened PID handles')
        return report

    def test_exited_history_is_not_killed_or_waited(self):
        r=self.cleanup_case(states=('gone',))
        self.assertEqual(r['processes'][0]['state'],'exited')
        self.assertEqual(r['unconfirmed_pids'],[])

    def test_reused_pid_is_not_killed_or_waited(self):
        r=self.cleanup_case(states=('reused',))
        self.assertEqual(r['processes'][0]['state'],'pid_reused')
        self.assertEqual(r['remaining_pids'],[])

    def test_access_denied_then_confirmed_exit(self):
        r=self.cleanup_case(states=('denied','gone'))
        self.assertEqual(r['processes'][0]['state'],'exited')
        self.assertEqual(r['unconfirmed_pids'],[])
        self.assertTrue(any(e['error_type']=='AccessDenied' for e in r['processes'][0]['events']))

    def test_access_denied_then_confirmed_pid_reuse(self):
        r=self.cleanup_case(states=('denied','reused'))
        self.assertEqual(r['processes'][0]['state'],'pid_reused')

    def test_access_denied_without_exit_evidence_remains_unconfirmed(self):
        r=self.cleanup_case(states=('denied',))
        self.assertEqual(r['unconfirmed_pids'],[987654321])

    def test_no_such_process_with_pid_still_present_is_unconfirmed(self):
        # psutil Windows can convert a permission error into NoSuchProcess.
        r=self.cleanup_case(states=('stale_nsp',))
        self.assertEqual(r['unconfirmed_pids'],[987654321])
        self.assertTrue(any(e['error_type']=='NoSuchProcess' for e in r['processes'][0]['events']))

    def test_wait_timeout_with_live_identity_is_not_reaped(self):
        r=self.cleanup_case()
        self.assertEqual(r['remaining_pids'],[987654321])
        self.assertEqual(r['processes'][0]['state'],'wait_timeout')

    def test_identity_exit_race_finishes_without_waiting_stale_handle(self):
        r=self.cleanup_case(states=('live','gone'))
        self.assertEqual(r['processes'][0]['state'],'exited')
        self.assertEqual(r['remaining_pids'],[])

    def test_missing_creation_time_does_not_authorize_kill(self):
        r=self.cleanup_case(created=None)
        self.assertEqual(r['unconfirmed_pids'],[987654321])

    def test_distinct_generations_of_one_pid_are_retained(self):
        known={}
        for created in (10.,20.):
            p=Mock(spec=psutil.Process);p.pid=55;p.create_time.return_value=created
            if not hasattr(supervised,'_remember_children'):self.fail('PID-only tracking loses historical process identities')
            supervised._remember_children(known,[p])
        self.assertEqual(set(known),{(55,10.),(55,20.)})


class CleanupStatusTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows signaled process handles')
    def test_terminated_process_object_is_not_a_live_orphan(self):
        # Retain a real Windows handle after termination. Reproduce the cloud
        # identity observation (same PID/create_time still visible) at its OS boundary.
        parent=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'])
        identity=psutil.Process(parent.pid);created=identity.create_time()
        try:
            self.assertFalse(supervised._windows_process_signaled(parent.pid))
            parent.terminate()
            parent.wait(timeout=10)
            original_process=psutil.Process;original_exists=psutil.pid_exists
            with patch.object(supervised.psutil,'pid_exists',side_effect=lambda pid: True if pid==parent.pid else original_exists(pid)), \
                 patch.object(supervised.psutil,'Process',side_effect=lambda pid: identity if pid==parent.pid else original_process(pid)):
                report=supervised._kill_tree(parent,{(parent.pid,created):identity},wait_timeout=.05)
            self.assertEqual(report['remaining_pids'],[],report)
            self.assertEqual(report['processes'][0]['state'],'exited',report)
            self.assertFalse(report['wait_timed_out'],report)
        finally:
            if parent.poll() is None:parent.kill()
            parent.wait(timeout=10)

    @unittest.skipUnless(os.name=='nt','Windows Job accounting')
    def test_job_accounting_failure_cannot_report_reaped(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('VRQO_OUTPUT_DIR',ROOT/'outputs')) as td:
            with patch.object(supervised._WindowsJob,'active_processes',side_effect=OSError('job accounting unavailable'),create=True):
                r=supervised.run_limited([sys.executable,'-B','-c','pass'],cwd=td,output_dir=Path(td)/'run',timeout_sec=None,memory_bytes=256*1024**2)
            self.assertEqual(r['returncode'],0)
            self.assertEqual(r['status'],'cleanup_failed')
            self.assertFalse(r['all_processes_reaped'])

    def test_unconfirmed_cleanup_keeps_successful_execution_separate(self):
        original=supervised._kill_tree
        def unresolved(*args,**kwargs):
            record=original(*args,**kwargs)
            if not isinstance(record,dict):self.fail('Cleanup outcomes must be propagated into process_result.json')
            record['unconfirmed_pids']=[987654321]
            return record
        with tempfile.TemporaryDirectory(dir=os.environ.get('VRQO_OUTPUT_DIR',ROOT/'outputs')) as td:
            with patch.object(supervised,'_kill_tree',side_effect=unresolved):
                r=supervised.run_limited([sys.executable,'-B','-c','pass'],cwd=td,output_dir=Path(td)/'run',timeout_sec=None,memory_bytes=256*1024**2)
            saved=json.loads((Path(td)/'run/process_result.json').read_text())
            self.assertEqual(r['returncode'],0)
            self.assertEqual(r.get('execution_status'),'passed')
            self.assertEqual(r['status'],'cleanup_failed')
            self.assertFalse(r['all_processes_reaped'])
            self.assertEqual(saved['unconfirmed_pids'],[987654321])


if __name__=='__main__':unittest.main()
