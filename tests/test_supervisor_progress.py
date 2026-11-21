"""Short control-path tests; no transport simulation or hour-long waits."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import supervised

ROOT=Path(__file__).resolve().parents[1]

class SupervisorProgressTests(unittest.TestCase):
    def folder(self):
        tmp=tempfile.TemporaryDirectory(dir=os.environ.get('VRQO_OUTPUT_DIR',ROOT/'outputs'))
        self.addCleanup(tmp.cleanup);return Path(tmp.name)

    def run_code(self,folder,code,**kwargs):
        return supervised.run_limited([sys.executable,'-B','-c',code],cwd=ROOT,output_dir=folder/'process',memory_bytes=256*1024**2,**kwargs)

    def test_registration_failure_is_reported_before_worker_launch(self):
        folder=self.folder();parent=folder/'parent.json'
        original=supervised._atomic_record
        def fail_registration(path,record):
            if Path(path).parent==parent.with_suffix(''):
                raise OSError('injected identity registration failure')
            return original(path,record)
        with patch.dict(os.environ,{'VRQO_PARENT_STATUS_FILE':str(parent)}),patch('supervised._atomic_record',side_effect=fail_registration),patch('supervised.subprocess.Popen') as launch:
            r=self.run_code(folder,'raise RuntimeError("must not execute")')
        launch.assert_not_called()
        self.assertEqual(r['status'],'supervisor_error')
        self.assertTrue(r['all_processes_reaped'])
        self.assertIn('injected identity registration failure',r['error'])
        self.assertEqual(json.loads((folder/'process/process_result.json').read_text())['status'],'supervisor_error')

    def test_display_publication_failure_warns_once_and_does_not_abort(self):
        folder=self.folder();stream=io.StringIO()
        with patch.dict(os.environ,{'VRQO_PROGRESS_FILE':str(folder/'progress.json')}),patch('supervised._atomic_record',side_effect=OSError('injected display failure')),contextlib.redirect_stderr(stream):
            supervised.publish_progress('starting')
            supervised.publish_progress('aer_submit',shots_completed=0,shots_total=300)
        self.assertEqual(stream.getvalue().count('[WARN]'),1)
        self.assertIn('injected display failure',stream.getvalue())
        code="import supervised;from unittest.mock import patch\nwith patch('supervised._atomic_record',side_effect=OSError('child display failure')):\n supervised.publish_progress('aer_submit',shots_completed=0,shots_total=300)\nprint('worker completed')"
        result=self.run_code(folder,code)
        self.assertEqual(result['status'],'passed',result)
        self.assertIn('worker completed',(folder/'process/stdout.log').read_text())
        self.assertIn('child display failure',(folder/'process/stderr.log').read_text())

    def test_process_creation_failure_is_reported_without_running_record(self):
        folder=self.folder();parent=folder/'parent.json'
        with patch.dict(os.environ,{'VRQO_PARENT_STATUS_FILE':str(parent)}),patch('supervised.subprocess.Popen',side_effect=OSError('injected CreateProcess failure')):
            result=self.run_code(folder,'pass')
        self.assertEqual(result['status'],'supervisor_error')
        self.assertTrue(result['all_processes_reaped'])
        self.assertIn('injected CreateProcess failure',result['error'])
        self.assertEqual(supervised._child_records(parent)[0]['status'],'supervisor_error')

    def test_display_recovery_reports_once_and_next_outage_warns_again(self):
        path=self.folder()/'progress.json';stream=io.StringIO()
        failure=PermissionError(13,'injected display outage');failure.winerror=5
        with contextlib.redirect_stderr(stream),contextlib.redirect_stdout(stream):
            for stage in ('first','second'):
                with patch('supervised._atomic_record',side_effect=failure):
                    self.assertFalse(supervised._display_record(path,{'stage':stage}))
                    self.assertFalse(supervised._display_record(path,{'stage':stage}))
                self.assertTrue(supervised._display_record(path,{'stage':'restored'}))
                self.assertTrue(supervised._display_record(path,{'stage':'restored'}))
        self.assertEqual(stream.getvalue().count('[WARN]'),2)
        self.assertEqual(stream.getvalue().count('[INFO]'),2)
        self.assertEqual(json.loads(path.read_text()),{'stage':'restored'})

    def test_failed_warning_stream_cannot_make_display_publication_fatal(self):
        class BrokenStream:
            def write(self,value):raise OSError('injected closed stderr')
            def flush(self):raise OSError('injected closed stderr')
        path=self.folder()/'progress.json'
        with patch('supervised.sys.stderr',BrokenStream()):
            with patch('supervised._atomic_record',side_effect=PermissionError('injected snapshot denial')):
                self.assertFalse(supervised._display_record(path,{'stage':'blocked'}))
            self.assertTrue(supervised._display_record(path,{'stage':'restored'}))
        self.assertEqual(json.loads(path.read_text()),{'stage':'restored'})

    def test_required_result_write_failure_cannot_report_success(self):
        folder=self.folder();original=supervised._atomic_record
        def deny_result(path,record):
            if Path(path).name=='process_result.json':raise OSError('injected required result failure')
            return original(path,record)
        with patch('supervised._atomic_record',side_effect=deny_result):
            with self.assertRaisesRegex(OSError,'injected required result failure'):
                self.run_code(folder,'pass')

    def test_resource_sampling_is_independent_of_poll_and_heartbeat(self):
        folder=self.folder();stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            r=self.run_code(folder,'import time;time.sleep(.55)',heartbeat_sec=.1,resource_sample_sec=.25,poll_sec=.02)
        self.assertEqual(r['status'],'passed',r)
        self.assertGreaterEqual(r['resource_sample_count'],2)
        self.assertLessEqual(r['resource_sample_count'],5)
        self.assertGreaterEqual(r['heartbeat_count'],3)
        resource=json.loads((folder/'process/resource.json').read_text())
        self.assertIn('cpu_interval_average_cores',resource)
        self.assertIn('processes',resource)
        self.assertIn('[HB]',stream.getvalue())
        self.assertNotIn('SUPERVISOR',stream.getvalue())

    def test_safety_samples_continue_between_resource_publications(self):
        folder=self.folder();writes=[];original=supervised._display_record
        def record(path,value):
            if Path(path).name=='resource.json':writes.append(value)
            return original(path,value)
        with patch('supervised._display_record',side_effect=record),contextlib.redirect_stdout(io.StringIO()):
            r=self.run_code(folder,'import time;time.sleep(.45)',heartbeat_sec=.2,
                            resource_sample_sec=.02,poll_sec=.01)
        self.assertEqual(r['status'],'passed',r)
        self.assertGreaterEqual(r['resource_sample_count'],10)
        self.assertLessEqual(len(writes),5,'Display publication must follow heartbeat, not safety polling')
        if os.name=='nt':self.assertGreater(r['peak_job_memory_bytes'],0)

    def test_default_deadline_is_absent_in_parent_and_worker(self):
        folder=self.folder()
        with patch.dict(os.environ):
            os.environ.pop('VRQO_DEADLINE_MONOTONIC',None)
            r=self.run_code(folder,"import os;print(repr(os.environ.get('VRQO_DEADLINE_MONOTONIC')))")
        self.assertEqual(r['status'],'passed')
        self.assertIsNone(r['timeout_seconds'])
        self.assertIsNone(r['effective_deadline_monotonic'])
        self.assertEqual((folder/'process/stdout.log').read_text().strip(),'None')

    def test_native_blocking_worker_does_not_block_supervisor_heartbeat(self):
        folder=self.folder()
        block="import ctypes;ctypes.PyDLL('kernel32').Sleep(450)" if os.name=='nt' else 'import time;time.sleep(.45)'
        code="from supervised import publish_progress;publish_progress('aer_sampling',shots_completed=0,shots_total=4096);"+block
        stream=io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):r=self.run_code(folder,code,heartbeat_sec=.05,echo_output=True)
        except TypeError as exc:self.fail(str(exc))
        self.assertEqual(r['status'],'passed',r)
        events=[json.loads(line) for line in (folder/'process/supervisor_heartbeat.jsonl').read_text().splitlines()]
        sampling=[e for e in events if e['stage']=='aer_sampling']
        self.assertGreaterEqual(len(sampling),3)
        self.assertGreaterEqual(sum(e['process_alive'] for e in sampling),3)
        self.assertTrue(all(e['shots_completed']==0 for e in sampling))
        self.assertTrue(all(e['elapsed_seconds']<r['elapsed_seconds'] for e in sampling))
        self.assertIn('[HB]',stream.getvalue())
        self.assertNotIn('SUPERVISOR',stream.getvalue())
        self.assertTrue(all(not e['stage_stale'] for e in sampling))

    def test_old_stage_age_is_reported_without_claiming_stall(self):
        folder=self.folder()
        code="import os,json,time;from pathlib import Path;p=Path(os.environ['VRQO_PROGRESS_FILE']);p.write_text(json.dumps({'stage':'compile','updated_monotonic':time.monotonic()-100}));time.sleep(.2)"
        try:r=self.run_code(folder,code,heartbeat_sec=.05)
        except TypeError as exc:self.fail(str(exc))
        self.assertEqual(r['status'],'passed',r)
        events=[json.loads(line) for line in (folder/'process/supervisor_heartbeat.jsonl').read_text().splitlines()]
        old=[e for e in events if e['stage']=='compile']
        self.assertTrue(old)
        self.assertTrue(all(not e['stage_stale'] and e['stage_age_seconds']>99 and e['shots_completed'] is None for e in old))

    def test_nested_timeout_keeps_cause_and_raw_parent_exit(self):
        folder=self.folder();inner=folder/'inner'
        code="from supervised import run_limited;import sys;run_limited([sys.executable,'-c','import time;time.sleep(20)'],cwd="+repr(str(ROOT))+",output_dir="+repr(str(inner))+",timeout_sec=.2,memory_bytes=200*1024**2);raise SystemExit(3)"
        r=self.run_code(folder,code)
        self.assertEqual(r['returncode'],3)
        self.assertEqual(r['execution_status'],'failed')
        self.assertEqual(r['status'],'timeout')
        self.assertEqual(r['termination_reason'],'child_timeout')
        self.assertEqual(r['child_process_report'],str(inner/'process_result.json'))
        self.assertTrue(r['all_processes_reaped'])

    def test_nested_interruption_and_memory_failure_are_preserved(self):
        for child_status,expected in [('interrupted','user_interrupted'),('memory_limit','memory_limit')]:
            with self.subTest(child_status=child_status):
                folder=self.folder()
                code="import os,json;from pathlib import Path;p=Path(os.environ['VRQO_PARENT_STATUS_FILE']);p.write_text(json.dumps({'status':"+repr(child_status)+",'process_result_path':'fault-injection','termination_reason':"+repr(expected)+"}));raise SystemExit(3)"
                r=self.run_code(folder,code)
                self.assertEqual(r['status'],child_status)
                self.assertEqual(r['termination_reason'],expected)
                self.assertEqual(r['returncode'],3)

    def test_successful_test_process_can_handle_expected_child_failure(self):
        folder=self.folder()
        code="import os,json;from pathlib import Path;Path(os.environ['VRQO_PARENT_STATUS_FILE']).write_text(json.dumps({'status':'timeout','process_result_path':'expected-fault'}))"
        r=self.run_code(folder,code)
        self.assertEqual(r['status'],'passed')
        self.assertEqual(r.get('termination_reason'),'completed')

    def test_later_child_success_does_not_erase_timeout_cause(self):
        folder=self.folder()
        code="from supervised import run_limited;import sys;root="+repr(str(ROOT))+";out="+repr(str(folder))+";run_limited([sys.executable,'-c','import time;time.sleep(5)'],cwd=root,output_dir=out+'/first',timeout_sec=.1,memory_bytes=200*1024**2);run_limited([sys.executable,'-c','pass'],cwd=root,output_dir=out+'/second',memory_bytes=200*1024**2);raise SystemExit(3)"
        r=self.run_code(folder,code)
        self.assertEqual(r['status'],'timeout')
        self.assertEqual(r['termination_reason'],'child_timeout')
        self.assertEqual(len(r['child_process_reports']),2)

    def test_finished_bridge_does_not_display_old_running_stage(self):
        folder=self.folder();leaf=folder/'leaf.json';top=folder/'top.json'
        leaf.write_text(json.dumps({'stage':'sampling','updated_monotonic':0}))
        top.write_text(json.dumps({'stage':'child_finished','child_progress_file':str(leaf)}))
        self.assertEqual(supervised._latest_progress(top)['stage'],'child_finished')

@unittest.skipUnless(os.name=='nt','Windows delete-sharing semantics')
class WindowsProgressSharingTests(unittest.TestCase):
    def test_transient_open_reader_does_not_abort_progress_publication(self):
        """Real OS failure: a heartbeat reader briefly denies atomic replacement."""
        with tempfile.TemporaryDirectory(prefix='paper1 progress sharing ') as td:
            root=Path(td);target=root/'outputs'/'run'/'progress.json'
            target.parent.mkdir(parents=True);target.write_text('{"stage":"old"}')
            held=target.open('r');replace=os.replace;errors=[]
            def release_after_first_sharing_error(source,destination):
                try:return replace(source,destination)
                except PermissionError as exc:
                    errors.append(exc.winerror);held.close();raise
            try:
                with patch('runtime_identity.PROJECT_ROOT',root),patch('supervised.os.replace',side_effect=release_after_first_sharing_error):
                    supervised._atomic_record(target,{'stage':'new'})
            finally:held.close()
            self.assertEqual(len(errors),1)
            self.assertIn(errors[0],(5,32,33))
            self.assertEqual(json.loads(target.read_text()),{'stage':'new'})
            self.assertEqual(list(target.parent.glob('.tmp-*')),[])

    def test_persistent_write_denial_still_fails_and_keeps_original(self):
        with tempfile.TemporaryDirectory(prefix='paper1 progress denial ') as td:
            root=Path(td);target=root/'outputs'/'run'/'progress.json'
            target.parent.mkdir(parents=True);target.write_text('{"stage":"old"}')
            denied=PermissionError('persistent sharing denial');denied.winerror=5
            with patch('runtime_identity.PROJECT_ROOT',root),patch('supervised.os.replace',side_effect=denied) as replace,patch('supervised.time.sleep'):
                with self.assertRaises(PermissionError):supervised._atomic_record(target,{'stage':'new'})
            self.assertGreater(replace.call_count,1)
            self.assertLessEqual(replace.call_count,10)
            self.assertEqual(json.loads(target.read_text()),{'stage':'old'})
            self.assertEqual(list(target.parent.glob('.tmp-*')),[])

if __name__=='__main__':unittest.main()
