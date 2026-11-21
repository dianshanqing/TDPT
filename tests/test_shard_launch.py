"""Windows launch/cleanup regressions; lightweight workers, no simulation."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil
import shard_runner
import supervised


ROOT = Path(__file__).resolve().parents[1]
ACK = """
import json, os, pathlib, time
ack = pathlib.Path(os.environ['VRQO_WORKER_STARTED_FILE'])
temporary_ack = ack.with_suffix('.tmp')
temporary_ack.write_text(json.dumps({'pid': os.getpid()}), encoding='utf8')
os.replace(temporary_ack, ack)
"""


@unittest.skipUnless(os.name == 'nt', 'Requires real Windows Job Objects')
class ShardLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/'outputs')
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.children = []
        self.original_popen = subprocess.Popen

    def popen(self, *args, **kwargs):
        child = self.original_popen(*args, **kwargs)
        self.children.append((child, psutil.Process(child.pid).create_time()))
        return child

    def coordinate(self, items, code, **kwargs):
        settings=dict(heartbeat_sec=0,resource_sample_sec=1,poll_sec=.02)
        settings.update(kwargs)
        return shard_runner.coordinate_shards(
            items, lambda item: [sys.executable, '-B', '-c', code, str(item['sid'])],
            out=self.out, cwd=ROOT, max_parallel=50, memory_bytes=4*1024**3,
            total_samples=len(items), **settings)

    def assert_reaped(self):
        for process, created in self.children:
            self.assertIsNotNone(process.poll(), f'Unreaped child {process.pid}')
            try:
                current = psutil.Process(process.pid)
                if current.create_time()==created:
                    self.assertTrue(supervised._windows_process_signaled(process.pid),
                                    f'Owned process {process.pid} survived cleanup')
            except psutil.NoSuchProcess:
                pass

    def test_fifty_workers_really_overlap_two_rounds_without_duplicates(self):
        # Every worker stays alive until all 50 have acknowledged startup.
        # No count inferred from submitted futures or opened log files.
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                self.out = Path(self.temporary.name)/f'round_{attempt}'
                barrier = self.out/'barrier'
                barrier.mkdir(parents=True)
                code = ACK + f"""
import sys
barrier = pathlib.Path({str(barrier)!r})
(barrier / (sys.argv[1] + '.json')).write_text(json.dumps({{'pid': os.getpid()}}))
deadline = time.monotonic() + 60
while len(list(barrier.glob('*.json'))) != 50:
    if time.monotonic() > deadline:
        raise RuntimeError('50-worker barrier was not reached')
    time.sleep(.05)
time.sleep(.3)
"""
                with patch('shard_runner.subprocess.Popen', side_effect=self.popen):
                    result = self.coordinate([{'sid': i, 'shots': 1} for i in range(50)], code)
                self.assertEqual(result['failures'], [], result)
                self.assertEqual(result['started_shards'], 50, result)
                self.assertEqual(result['peak_running'], 50, result)
                self.assertEqual(len(result['states']), 50)
                self.assertEqual({r['sid'] for r in result['states']}, set(range(50)))
                self.assertTrue(all(r['state'] == 'complete' for r in result['states']), result)
                self.assertTrue(all(r['history'] == ['pending', 'starting', 'started', 'running', 'complete']
                                    for r in result['states']), result)
                markers = [json.loads(p.read_text()) for p in barrier.glob('*.json')]
                self.assertEqual(len({m['pid'] for m in markers}), 50)
                self.assertTrue(result['all_processes_reaped'], result)
                self.assert_reaped()

    def test_duplicate_shard_id_is_rejected_before_popen(self):
        with patch('shard_runner.subprocess.Popen') as popen:
            with self.assertRaises(ValueError):
                self.coordinate([{'sid': 4, 'shots': 1}, {'sid': 4, 'shots': 1}], ACK)
        popen.assert_not_called()

    def test_stale_resource_sample_is_unavailable_not_current_activity(self):
        resource=self.out/'resource.json'
        resource.write_text(json.dumps({'updated_monotonic':0,
            'batch_cpu_interval_average_cores':50,'batch_rss_mib':90000}),encoding='utf8')
        capture=io.StringIO()
        with patch.dict(os.environ,{'VRQO_RESOURCE_FILE':str(resource)}),contextlib.redirect_stdout(capture):
            result=shard_runner.coordinate_shards([{'sid':0,'shots':1}],
                lambda item:[sys.executable,'-B','-c',ACK+'time.sleep(.3)'],
                out=self.out,cwd=ROOT,max_parallel=50,memory_bytes=4*1024**3,
                heartbeat_sec=.1,resource_sample_sec=.05,poll_sec=.02,total_samples=1)
        self.assertEqual(result['failures'],[])
        self.assertIn('CPU=unavailable',capture.getvalue())
        self.assertNotIn('CPU=50.00',capture.getvalue())

    def test_popen_failure_is_immediate_and_reaps_already_owned_workers(self):
        calls = 0
        def fail_third(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError('injected process creation failure')
            return self.popen(*args, **kwargs)
        capture = io.StringIO()
        started = time.monotonic()
        with patch('shard_runner.subprocess.Popen', side_effect=fail_third), \
                contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            result = self.coordinate([{'sid': i, 'shots': 1} for i in range(5)], ACK+'time.sleep(30)\n')
        self.assertLess(time.monotonic()-started, 10, result)
        self.assertEqual(calls, 3, result)
        self.assertTrue(result['failures'], result)
        self.assertEqual(result['states'][2]['state'], 'failed', result)
        self.assertTrue(all(r['state'] == 'pending' for r in result['states'][3:]), result)
        self.assertIn('injected process creation failure', capture.getvalue())
        self.assertIn('002', capture.getvalue())
        self.assertTrue(result['all_processes_reaped'], result)
        self.assert_reaped()

    def test_exit_zero_without_worker_ack_cannot_be_success(self):
        with patch('shard_runner.subprocess.Popen', side_effect=self.popen):
            result = self.coordinate([{'sid': 6, 'shots': 1}], 'pass')
        self.assertTrue(result['failures'], result)
        self.assertEqual(result['states'][0]['state'], 'failed', result)
        self.assertEqual(result['started_shards'], 0, result)
        self.assertTrue(result['all_processes_reaped'], result)
        self.assert_reaped()

    def test_native_worker_exit_keeps_healthy_shards_and_reports_immediately(self):
        # Reproduce the T15 exit status, without a real crash or a WER dialog.
        code=ACK+"""
import ctypes, sys, faulthandler
sid=int(sys.argv[1])
if sid==1:(ack.parent/'fault_handler').write_text(str(faulthandler.is_enabled()))
deadline=time.monotonic()+15
while len(list(ack.parent.glob('*.started.json')))<3:
    if time.monotonic()>deadline:raise RuntimeError('startup barrier timeout')
    time.sleep(.02)
if sid==1:ctypes.windll.kernel32.ExitProcess(0xC0000005)
# Healthy workers finish only after the coordinator has reported the failure.
while not (ack.parent/'error_observed').exists():
    if time.monotonic()>deadline:raise RuntimeError('failure was not reported promptly')
    time.sleep(.02)
(ack.parent/f'completed_{sid}').write_text('done')
"""
        class Capture(io.StringIO):
            def write(stream,text):
                if '[ERROR] shard=1 ' in text:
                    (self.out/'launch_logs/error_observed').write_text('seen')
                return super().write(text)
        capture=Capture()
        with patch('shard_runner.subprocess.Popen',side_effect=self.popen),contextlib.redirect_stdout(capture):
            result=self.coordinate([{'sid':sid,'shots':1} for sid in range(3)],code)
        self.assertEqual([r['state'] for r in result['states']],['complete','failed','complete'],result)
        self.assertEqual(len(result['failures']),1,result)
        self.assertIn('0xC0000005',capture.getvalue())
        self.assertEqual(result['states'][1]['returncode'],0xC0000005)
        self.assertEqual(len(self.children),3,'No automatic retry')
        self.assertEqual((self.out/'launch_logs/fault_handler').read_text(),'True')
        self.assertTrue(result['all_processes_reaped'],result)
        for sid in (0,2):self.assertTrue((self.out/f'launch_logs/completed_{sid}').exists())
        self.assert_reaped()

    def test_worker_identity_write_failure_is_visible_and_not_started(self):
        capture = io.StringIO()
        with patch('shard_runner.subprocess.Popen', side_effect=self.popen), \
                contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            result = self.coordinate([{'sid': 14, 'shots': 1}],
                                     "raise OSError('injected worker identity write failure')")
        self.assertTrue(result['failures'], result)
        self.assertEqual(result['started_shards'], 0, result)
        self.assertEqual(result['states'][0]['state'], 'failed', result)
        self.assertIn('014', capture.getvalue())
        self.assertTrue(result['all_processes_reaped'], result)
        self.assert_reaped()

    def test_periodic_snapshot_replace_denial_keeps_workers_alive_and_recovers(self):
        """Reproduce the cloud failure after startup, through real atomic-write retries."""
        original_replace=os.replace;denied_attempts=0;previous=None;stable_seen=False;held=None;errors=[]
        finish=self.out/'allow_finish';target=self.out/'batch_state.json'
        def deny_one_publication(source,destination):
            nonlocal denied_attempts,previous,stable_seen,held
            if Path(destination)==target:
                value=json.loads(Path(source).read_text(encoding='utf8'))
                computing=all(r['state']=='running' and r.get('stage')=='aer_submit' for r in value['states'])
                if computing and not stable_seen:
                    stable_seen=True
                    return original_replace(source,destination)
                if computing and denied_attempts<9:
                    if previous is None:
                        previous=target.read_bytes()
                        held=target.open('r')  # Actual Windows handle denies delete sharing.
                    self.assertEqual(target.read_bytes(),previous,'A failed replacement altered the valid snapshot')
                    try:return original_replace(source,destination)
                    except PermissionError as exc:
                        errors.append(exc.winerror);denied_attempts+=1
                        if denied_attempts==9:
                            held.close();held=None;finish.touch()
                        raise
            return original_replace(source,destination)
        code=ACK+f"""
progress=pathlib.Path(os.environ['VRQO_PROGRESS_FILE'])
temporary_progress=progress.with_suffix('.tmp')
temporary_progress.write_text(json.dumps({{'stage':'aer_submit','shots_completed':0,'shots_total':1}}))
os.replace(temporary_progress,progress)
deadline=time.monotonic()+10
while not pathlib.Path({str(finish)!r}).exists():
    if time.monotonic()>deadline:raise RuntimeError('snapshot fault was not exercised')
    time.sleep(.02)
time.sleep(.15)
"""
        capture = io.StringIO()
        try:
            with patch('shard_runner.subprocess.Popen', side_effect=self.popen), \
                    patch('supervised.os.replace', side_effect=deny_one_publication), \
                    contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                result=self.coordinate([{'sid':i,'shots':1} for i in range(2)],code,
                                       heartbeat_sec=.1,resource_sample_sec=.03)
        finally:
            if held:held.close()
        self.assertEqual(denied_attempts,9)
        self.assertTrue(all(error in (5,32,33) for error in errors),errors)
        self.assertEqual(result['failures'],[],result)
        self.assertTrue(all(r['state']=='complete' and r['returncode']==0 for r in result['states']),result)
        self.assertEqual(result['snapshot_status'],'current')
        self.assertEqual(result['snapshot_write_failures'],1)
        self.assertEqual(json.loads(target.read_text())['states'],result['states'])
        self.assertEqual(capture.getvalue().count('[WARN]'),1)
        self.assertEqual(capture.getvalue().count('[INFO]'),1)
        self.assertIn('state_snapshot=unavailable',capture.getvalue())
        self.assertEqual(list(self.out.glob('.tmp-*')),[])
        self.assertTrue(result['all_processes_reaped'], result)
        self.assert_reaped()

    def test_snapshot_denial_through_completion_is_visible_without_science_failure(self):
        original_replace=os.replace;denied_attempts=0;previous=None;stable_seen=False
        finish=self.out/'allow_finish';target=self.out/'batch_state.json'
        def deny_running_snapshot(source,destination):
            nonlocal denied_attempts,previous,stable_seen
            if Path(destination)==target:
                value=json.loads(Path(source).read_text(encoding='utf8'))
                computing=all(r['state']=='running' and r.get('stage')=='aer_submit' for r in value['states'])
                if computing and not stable_seen:
                    stable_seen=True
                    return original_replace(source,destination)
                if (stable_seen and computing) or denied_attempts:
                    if previous is None:previous=target.read_bytes()
                    self.assertEqual(target.read_bytes(),previous)
                    denied_attempts+=1
                    if denied_attempts==9:finish.touch()
                    failure=PermissionError(13,'injected persistent snapshot replacement denial')
                    failure.winerror=5
                    raise failure
            return original_replace(source,destination)
        code=ACK+f"""
progress=pathlib.Path(os.environ['VRQO_PROGRESS_FILE'])
temporary_progress=progress.with_suffix('.tmp')
temporary_progress.write_text(json.dumps({{'stage':'aer_submit','shots_completed':0,'shots_total':1}}))
os.replace(temporary_progress,progress)
deadline=time.monotonic()+10
while not pathlib.Path({str(finish)!r}).exists():
    if time.monotonic()>deadline:raise RuntimeError('snapshot fault was not exercised')
    time.sleep(.02)
time.sleep(.15)
"""
        capture=io.StringIO()
        with patch('shard_runner.subprocess.Popen',side_effect=self.popen), \
                patch('supervised.os.replace',side_effect=deny_running_snapshot), \
                contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
            result=self.coordinate([{'sid':i,'shots':1} for i in range(2)],code,
                                   heartbeat_sec=.1,resource_sample_sec=.03)
        self.assertGreaterEqual(denied_attempts,18)
        self.assertEqual(result['failures'],[],result)
        self.assertTrue(all(r['state']=='complete' and r['returncode']==0 for r in result['states']),result)
        self.assertEqual(result['snapshot_status'],'unavailable')
        self.assertEqual(result['snapshot_write_failures'],denied_attempts//9)
        self.assertEqual(target.read_bytes(),previous)
        self.assertEqual(capture.getvalue().count('[WARN]'),1)
        self.assertNotIn('[INFO]',capture.getvalue())
        self.assertIn('state_snapshot=unavailable',capture.getvalue())
        self.assertTrue(result['all_processes_reaped'],result)
        self.assert_reaped()

    def test_stable_snapshot_does_not_follow_fast_resource_sampling(self):
        """The worker observes its status file while running; no repeated file churn."""
        observations=self.out/'snapshot_observations.json';target=self.out/'batch_state.json'
        code=ACK+f"""
target=pathlib.Path({str(target)!r})
deadline=time.monotonic()+10
while True:
    if time.monotonic()>deadline:raise RuntimeError('running state was not published')
    try:
        state=json.loads(target.read_text())['states'][0]['state']
        if state=='running':break
    except (OSError,ValueError):pass
    time.sleep(.02)
stamps=[]
for unused in range(10):
    stamps.append(target.stat().st_mtime_ns)
    time.sleep(.04)
pathlib.Path({str(observations)!r}).write_text(json.dumps(stamps))
"""
        with patch('shard_runner.subprocess.Popen',side_effect=self.popen), \
                patch('supervised._latest_progress',wraps=supervised._latest_progress) as samples:
            result=self.coordinate([{'sid':0,'shots':1}],code,resource_sample_sec=.02)
        self.assertEqual(result['failures'],[],result)
        self.assertLessEqual(samples.call_count,1,'Display reads must also follow heartbeat cadence')
        self.assertEqual(len(set(json.loads(observations.read_text()))),1)
        self.assertEqual(result['snapshot_interval_sec'],60.)
        self.assertEqual(result['snapshot_write_failures'],0)
        self.assertTrue(result['all_processes_reaped'],result)
        self.assert_reaped()

    def test_keyboard_interrupt_reaps_owned_workers(self):
        sleep = time.sleep
        injected = False
        descendants = self.out/'descendants'
        descendants.mkdir()
        def interrupt_once(seconds):
            nonlocal injected
            if len(self.children) >= 4 and list(descendants.glob('*.json')) and not injected:
                injected = True
                raise KeyboardInterrupt()
            return sleep(seconds)
        code = ACK + f"""
import subprocess, sys, psutil
child = subprocess.Popen([sys.executable, '-B', '-c', 'import time;time.sleep(30)'])
record=pathlib.Path({str(descendants)!r},sys.argv[1]+'.json')
temporary_record=record.with_suffix('.tmp')
temporary_record.write_text(json.dumps({{'pid':child.pid,'create_time':psutil.Process(child.pid).create_time()}}))
os.replace(temporary_record,record)
time.sleep(30)
"""
        with patch('shard_runner.subprocess.Popen', side_effect=self.popen), \
                patch('shard_runner.time.sleep', side_effect=interrupt_once):
            result = self.coordinate([{'sid': i, 'shots': 1} for i in range(5)], code)
        self.assertTrue(injected)
        self.assertTrue(result['failures'], result)
        self.assertTrue(result['all_processes_reaped'], result)
        self.assert_reaped()
        for path in descendants.glob('*.json'):
            identity = json.loads(path.read_text())
            try:
                current = psutil.Process(identity['pid'])
                if current.create_time()==identity['create_time']:
                    self.assertTrue(supervised._windows_process_signaled(identity['pid']),
                                    f'Owned descendant {identity["pid"]} survived interruption')
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    unittest.main()
