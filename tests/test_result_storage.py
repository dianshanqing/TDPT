"""Lossless storage contracts, including legacy references and summary consumers."""
from contextlib import redirect_stdout
from dataclasses import dataclass
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch
import zlib

import runtime_identity


class ResultStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root/'outputs'/'run'
        self.output.mkdir(parents=True)
        self.root_patch = patch.object(runtime_identity, 'PROJECT_ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def storage(self):
        self.assertIsNotNone(importlib.util.find_spec('result_storage'),
                             'A transparent lossless result storage module is required')
        return importlib.import_module('result_storage')

    def test_streaming_json_matches_existing_bytes_and_keeps_small_result_readable(self):
        storage = self.storage()
        @dataclass
        class Settings:
            z: tuple
            a: Path
        payload = {'z': Settings((1, -0.0, 1e-14, complex(2, 3)), Path('model')),
                   'unicode': '数据\\\"\t\r\n' + 'x'*65530 + '\"\n数据',
                   1: 'replaced', '1': 'kept', 'nested': [[], {}, True, None]}
        expected = json.dumps(runtime_identity.normalize(payload), ensure_ascii=False,
                              indent=2, allow_nan=False).replace('\n', os.linesep).encode('utf-8')
        path = self.output/'small.json'
        actual = runtime_identity.write_json_new(path, payload)
        self.assertEqual(actual, path)
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(storage.read_json(path), json.loads(expected))
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ['small.json'])

    def test_auto_compression_preserves_json_bytes_and_old_logical_path(self):
        storage = self.storage()
        payload = {'raw_log': 'step 数据\n'*2000, 'probability': 1e-23}
        expected = json.dumps(runtime_identity.normalize(payload), ensure_ascii=False,
                              indent=2, allow_nan=False).replace('\n', os.linesep).encode('utf-8')
        logical = self.output/'large.json'
        with patch.object(storage, 'AUTO_COMPRESS_MIN_BYTES', 1024):
            physical = runtime_identity.write_json_new(logical, payload)
        self.assertEqual(physical, Path(str(logical)+'.gz'))
        self.assertFalse(logical.exists())
        self.assertEqual(gzip.decompress(physical.read_bytes()), expected)
        self.assertEqual(storage.read_json(logical), payload)
        self.assertEqual(storage.read_json(physical), payload)
        record = storage.storage_record(logical)
        self.assertEqual(record['original_sha256'], hashlib.sha256(expected).hexdigest())
        self.assertEqual(record['original_bytes'], len(expected))
        self.assertEqual(record['stored_sha256'], hashlib.sha256(physical.read_bytes()).hexdigest())
        self.assertEqual(record['stored_bytes'], physical.stat().st_size)
        self.assertEqual(record['compression'], 'gzip')

    def test_existing_either_spelling_refuses_both_new_destinations(self):
        storage = self.storage()
        for existing_suffix in ('.json', '.json.gz'):
            for requested_suffix in ('.json', '.json.gz'):
                with self.subTest(existing=existing_suffix, requested=requested_suffix):
                    path = self.output/('result'+existing_suffix)
                    path.write_bytes(b'existing evidence')
                    try:
                        with self.assertRaises(FileExistsError):
                            runtime_identity.assert_new_file(self.output/('result'+requested_suffix))
                        with self.assertRaises(FileExistsError):
                            runtime_identity.write_json_new(self.output/('result'+requested_suffix), {'x': 1})
                        self.assertEqual(path.read_bytes(), b'existing evidence')
                    finally:
                        path.unlink()
        self.assertFalse(list(self.output.iterdir()))

    def test_failed_serialization_leaves_no_final_or_partial_output(self):
        storage = self.storage()
        for payload in ({'later': object()}, {'bad': float('nan')}):
            with self.assertRaises((TypeError, ValueError)):
                runtime_identity.write_json_new(self.output/'bad.json', payload)
            self.assertEqual(list(self.output.iterdir()), [])
        with self.assertRaises(ValueError):
            runtime_identity.write_json_new(self.root/'outside.json', {'x': 1})

    def test_historical_compression_preserves_unusual_bytes_before_verified_unlink(self):
        storage = self.storage()
        original = b'\xef\xbb\xbf{\r\n  "z" : -0.0, "a":1e-023, "z":null\r\n}\n\t'
        logical = self.output/'historical.json'
        logical.write_bytes(original)
        record = storage.compress_file_lossless(logical)
        physical = Path(str(logical)+'.gz')
        self.assertEqual(logical.read_bytes(), original)
        self.assertEqual(gzip.decompress(physical.read_bytes()), original)
        self.assertTrue(record['verified'])
        self.assertFalse(record['removed_original'])
        self.assertEqual(record['original_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(storage.verify_lossless_copy(logical)['original_bytes'], len(original))
        removed = storage.remove_verified_original(logical)
        self.assertTrue(removed['removed_original'])
        self.assertFalse(logical.exists())
        with storage.open_result(logical, 'rb') as handle:
            self.assertEqual(handle.read(), original)

    def test_corrupt_or_truncated_gzip_never_removes_original_or_reads_as_plain(self):
        storage = self.storage()
        logical = self.output/'data.log'
        original = b'raw log\r\n'*1000
        logical.write_bytes(original)
        compressed = gzip.compress(original)
        physical = Path(str(logical)+'.gz')
        for bad in (compressed[:-5], compressed[:20]+bytes([compressed[20]^255])+compressed[21:]):
            with self.subTest(size=len(bad)):
                physical.write_bytes(bad)
                with self.assertRaises((OSError, EOFError, ValueError, zlib.error)):
                    storage.remove_verified_original(logical)
                self.assertEqual(logical.read_bytes(), original)
                with self.assertRaises((OSError, EOFError, ValueError, zlib.error)):
                    with storage.open_result(physical, 'rb') as handle:
                        handle.read()
        physical.write_bytes(gzip.compress(b'different'))
        with self.assertRaises(ValueError):
            storage.remove_verified_original(logical)
        self.assertEqual(logical.read_bytes(), original)

    def test_source_change_after_compression_blocks_verified_removal(self):
        storage = self.storage()
        logical = self.output/'live.log'
        logical.write_bytes(b'initial\r\n')
        storage.compress_file_lossless(logical)
        with logical.open('ab') as handle:
            handle.write(b'later\r\n')
        with self.assertRaises(ValueError):
            storage.remove_verified_original(logical)
        self.assertEqual(logical.read_bytes(), b'initial\r\nlater\r\n')

    def test_finalized_log_keeps_small_live_log_and_compresses_closed_large_log(self):
        storage = self.storage()
        logical = self.output/'stdout.log'
        logical.write_bytes(b'first\r\n')
        with patch.object(storage, 'AUTO_COMPRESS_MIN_BYTES', 1024):
            self.assertEqual(storage.finalize_log(logical), logical)
            with logical.open('ab') as handle:
                handle.write(b'next\r\n'*1000)
            original = logical.read_bytes()
            physical = storage.finalize_log(logical)
        self.assertFalse(logical.exists())
        self.assertEqual(gzip.decompress(physical.read_bytes()), original)

    def test_streaming_large_string_has_bounded_additional_python_memory(self):
        storage = self.storage()
        payload = {'MPS_log_data': 'raw\\\"\n数据 '*700000}
        tracemalloc.start()
        try:
            runtime_identity.write_json_new(self.output/'large.json', payload)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 8*1024*1024, 'Serialization must not duplicate the entire raw log')
        self.assertEqual(storage.read_json(self.output/'large.json'), payload)

    def test_probe_log_reference_preserves_original_hash_after_gzip(self):
        storage = self.storage()
        import validation
        row = {'status': 'complete', 'probabilities': [1., 1e-23, 0., 0.],
               'metadata': {'MPS_log_data': '数据 gate\n'*1000, 'method': 'matrix_product_state'}}
        logical = self.output/'case.json'
        runtime_identity.write_json_new(logical, row)
        original_bytes = logical.read_bytes()
        original_summary = validation._probe_summary_row(row, logical)
        storage.compress_file_lossless(logical, remove_original=True)
        summary = validation._probe_summary_row(row, logical)
        ref = summary['metadata_log_references']['MPS_log_data']
        self.assertEqual(ref['path'], 'case.json')
        self.assertEqual(ref['file_sha256'], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(ref['file_sha256'], original_summary['metadata_log_references']['MPS_log_data']['file_sha256'])
        self.assertEqual(ref['file_bytes'], len(original_bytes))
        self.assertEqual(ref['storage']['path'], 'case.json.gz')
        self.assertEqual(summary['probabilities'], row['probabilities'])

    def test_actual_probe_summarize_equivalent_for_directory_logical_and_gzip(self):
        storage = self.storage()
        import run
        payload = {'rows': [{'case': {'id': 'tiny'}, 'status': 'complete', 'bond': 2,
                   'truncation_threshold': 0., 'probabilities': [1.-1e-10, 1e-10, 0., 0.],
                   'reference': [1.-1e-10, 1e-10, 0., 0.],
                   'work_any_nonzero_union_upper_bound': 0., 'work_bit_probabilities': [0.]}],
                   'counts': []}
        logical = self.output/'mps_probability_probe.json'
        runtime_identity.write_json_new(logical, payload)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run.summarize(['--input', str(logical), '--outdir', str(self.root/'outputs'/'before')]), 0)
        before = storage.read_json(self.root/'outputs'/'before'/'result.json')
        storage.compress_file_lossless(logical, remove_original=True)
        for index, source in enumerate((logical, Path(str(logical)+'.gz'), logical.parent)):
            out = self.root/'outputs'/('after'+str(index))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run.summarize(['--input', str(source), '--outdir', str(out)]), 0)
            after = storage.read_json(out/'result.json')
            for key in ('rows', 'counts', 'execution_status', 'source_sha256', 'interpretation'):
                self.assertEqual(after[key], before[key], key)

    def test_actual_mc_summary_uses_gzip_shards_with_identical_statistics(self):
        storage = self.storage()
        import run
        identity = runtime_identity.make_identity(config={}, profile=None, compile_options=None,
                     measurement=None, backend={'method': 'numpy_mc'}, phase='mc', extra={'steps': 2})
        raw = {'phase': 'mc', 'status': 'complete', 'run_id': 'storage-fixture', 'request_hash': 'same',
               'identity': identity, 'steps': 2, 'histories': 10,
               'counts': {'survive': 4, 'detect': 3, 'absorb': 2, 'boundary': 1},
               'sampling': {'sample_start': 0, 'sample_stop': 10, 'total_samples': 10, 'num_shards': 1,
                   'shard_id': 0, 'root_seed': 71, 'actual_seed': 71, 'seed_strategy': 'root_seed + shard_id'}}
        logical = self.output/'shard_000'/'results.json'
        runtime_identity.write_json_new(logical, raw)
        with redirect_stdout(io.StringIO()):
            run.summarize(['--input', str(logical.parent.parent), '--outdir', str(self.root/'outputs'/'before')])
        before = storage.read_json(self.root/'outputs'/'before'/'result.json')
        storage.compress_file_lossless(logical, remove_original=True)
        with redirect_stdout(io.StringIO()):
            run.summarize(['--input', str(logical.parent.parent), '--outdir', str(self.root/'outputs'/'after')])
        after = storage.read_json(self.root/'outputs'/'after'/'result.json')
        self.assertEqual(before, after)

    def test_actual_small_mps_auto_gzip_and_batch_original_hash_validation(self):
        storage = self.storage()
        from batch_runtime import validate_task_result
        from run_request import resolve_run_request
        from validation import run_mps_probe
        folder = self.output/'tasks'/'tiny'
        request = resolve_run_request(dict(task='mps_probe', config='q_mps_one_cell', steps=1,
            profile='compressed', bond=16, mps_trunc=1e-16, shots=2, counts_check=False,
            cpu_budget=1, aer_threads=1, mps_threads=1, blas_threads=1, seed=91,
            outdir=str(folder)), project_root=self.root)
        # Lower only the storage threshold: the one-step transport is real.
        with patch.object(storage, 'AUTO_COMPRESS_MIN_BYTES', 1024), patch.dict(os.environ,
                {'VRQO_PROGRESS_FILE': str(self.output/'progress'/'progress.json')}), redirect_stdout(io.StringIO()):
            runtime_identity.write_json_new(folder/'resolved_request.json', request)
            report = run_mps_probe(request['effective_suite'], folder, cpu_budget=1,
                                   aer_threads=1, mps_threads=1, blas_threads=1)
        self.assertEqual(report['execution_status'], 'passed')
        reference = report['rows'][0]['metadata_log_references']['MPS_log_data']
        original_path = folder/reference['path']
        self.assertFalse(original_path.exists())
        self.assertTrue(Path(str(original_path)+'.gz').is_file())
        raw = storage.read_json(original_path)
        self.assertEqual(reference['file_sha256'], storage.content_digest(original_path)['sha256'])
        self.assertNotEqual(reference['file_sha256'], hashlib.sha256(Path(str(original_path)+'.gz').read_bytes()).hexdigest())
        actual = validate_task_result({'request': request}, raw['identity']['sources'], raw['identity']['versions'])
        self.assertEqual(actual['probabilities'], raw['probabilities'])
        # A bad legacy hash must still fail even when compression is valid.
        summary_path = storage.resolve_result_path(folder/'mps_probability_probe.json')
        original_summary = summary_path.read_bytes()
        changed = storage.read_json(summary_path)
        changed['rows'][0]['metadata_log_references']['MPS_log_data'].pop('storage')
        changed['rows'][0]['metadata_log_references']['MPS_log_data']['file_sha256'] = '0'*64
        try:
            summary_path.write_bytes(gzip.compress(json.dumps(changed).encode('utf-8')))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                validate_task_result({'request': request}, raw['identity']['sources'], raw['identity']['versions'])
        finally:
            summary_path.write_bytes(original_summary)

    def test_generic_analysis_readers_and_reference_baseline_read_old_gzip_paths(self):
        storage = self.storage()
        import analyze_alignment_paper1
        import compare_mc_test
        import mps_performance
        import postprocess_alignment
        raw = {'execution_status': 'passed', 'probabilities': [1., 0., 0., 0.],
               'metadata': {'MPS_log_data': 'retained raw bytes'}, 'p_detect': 0.}
        logical = self.output/'results.json'
        runtime_identity.write_json_new(logical, raw)
        before = storage.content_digest(logical)['sha256']
        storage.compress_file_lossless(logical, remove_original=True)
        self.assertEqual(analyze_alignment_paper1._load_results(self.output), raw)
        self.assertEqual(compare_mc_test._read_json(str(logical)), raw)
        self.assertEqual(postprocess_alignment._load_mc_payload(str(self.output))['probabilities'], raw['probabilities'])
        baseline = self.output/'performance_summary.json'
        runtime_identity.write_json_new(baseline, {'record_type': 'performance_summary',
                    'result_path': logical.name, 'result_file_sha256': before})
        storage.compress_file_lossless(baseline, remove_original=True)
        restored, provenance = mps_performance._load_reference(baseline)
        self.assertEqual(restored['probabilities'], raw['probabilities'])
        self.assertEqual(provenance['sha256'], before)
        self.assertNotIn('MPS_log_data', restored['metadata'])

    def test_compression_refuses_existing_archive_and_trailer_failure_keeps_source(self):
        storage = self.storage()
        logical = self.output/'source.log'
        logical.write_bytes(b'preserved')
        archive = Path(str(logical)+'.gz')
        archive.write_bytes(b'old archive')
        with self.assertRaises(FileExistsError):
            storage.compress_file_lossless(logical, remove_original=True)
        self.assertEqual(logical.read_bytes(), b'preserved')
        self.assertEqual(archive.read_bytes(), b'old archive')

    def test_simultaneous_new_writers_publish_one_complete_representation(self):
        storage = self.storage()
        from concurrent.futures import ThreadPoolExecutor
        import threading
        barrier = threading.Barrier(2)
        logical = self.output/'race.json'
        def submit(path, value):
            barrier.wait()
            try:
                return runtime_identity.write_json_new(path, {'winner': value, 'data': value*1000})
            except FileExistsError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit, path, value) for path, value in
                       ((logical, 'plain'), (Path(str(logical)+'.gz'), 'compressed'))]
            results = [future.result() for future in futures]
        self.assertEqual(sum(path is not None for path in results), 1)
        self.assertEqual(len(list(self.output.iterdir())), 1)
        actual = storage.read_json(logical)
        self.assertIn(actual['winner'], ('plain', 'compressed'))
        self.assertEqual(actual['data'], actual['winner']*1000)

    def test_archived_postprocess_discovers_and_reads_gzip_with_unchanged_rows(self):
        storage = self.storage()
        from tools import postprocess_archived_highbond_rare_response as archive
        from tools import postprocess_rare_response_mps as analysis
        folder = self.output/'test_s12_shards_total10_sumonly_seed1_chunk2_16_1e-16'
        merged = folder/'merged'/'results_merged.json'
        runtime_identity.write_json_new(merged, {'p_detect': .1, 'se_detect': .01, 'shots': 10, 'steps': 12})
        exact = self.output/'exact.json'
        runtime_identity.write_json_new(exact, {'final': {'p_detect_reach': .1}})
        targets = self.output/'targets.csv'
        targets.write_text('steps,p_target,eta,feasible,p_rare_exact\n12,0.001,0.01,true,0.001\n', encoding='utf-8')
        before = analysis.postprocess(target_csv=str(targets), mps_merged_json=str(merged), exact_json=str(exact))
        storage.compress_file_lossless(merged, remove_original=True)
        storage.compress_file_lossless(exact, remove_original=True)
        paths = archive._discover(self.output, {12})
        self.assertEqual([storage.logical_path(path) for path in paths], [merged])
        self.assertEqual(archive._read_json(paths[0])['p_detect'], .1)
        after = analysis.postprocess(target_csv=str(targets), mps_merged_json=str(merged), exact_json=str(exact))
        self.assertEqual(after, before)

    def test_validation_observation_collector_reads_finalized_jsonl_and_json(self):
        storage = self.storage()
        from run_local_validation import _scientific_observations
        folder = self.root/'outputs'/'_validation_groups'/'group'
        folder.mkdir(parents=True)
        journal = folder/'transport_science.jsonl'
        journal.write_bytes(b'{"probability":1e-23}\r\n{"probability":0.0}\r\n')
        quantum = folder/'quantum_ae_fixture.json'
        runtime_identity.write_json_new(quantum, {'profile': 'compressed', 'steps': 1,
            'A_fingerprint': 'same', 'inverse_amplitude_error': 0., 'inverse_tail_probability': 0.,
            'probability_tolerance': 1e-12, 'checks': [{'target': 'detect', 'iterations': 1, 'all_registers_reflected': True}]})
        before = _scientific_observations(self.root)
        storage.compress_file_lossless(journal, remove_original=True)
        storage.compress_file_lossless(quantum, remove_original=True)
        self.assertEqual(_scientific_observations(self.root), before)

    def test_supervisor_defers_compression_when_process_exit_is_unconfirmed(self):
        storage = self.storage()
        import supervised
        original_cleanup = supervised._kill_tree
        def unconfirmed(*args, **kwargs):
            result = original_cleanup(*args, **kwargs)
            return {**result, 'job_confirmed_empty': False}
        folder = self.output/'unconfirmed'
        with patch.object(storage, 'AUTO_COMPRESS_MIN_BYTES', 1024), patch.object(
                supervised, '_kill_tree', side_effect=unconfirmed), patch.dict(os.environ):
            for key in ('VRQO_PROGRESS_FILE', 'VRQO_PARENT_STATUS_FILE', 'VRQO_DEADLINE_MONOTONIC'):
                os.environ.pop(key, None)
            result = supervised.run_limited([sys.executable, '-B', '-c', 'import sys;sys.stdout.write("x"*4096)'],
                cwd=self.root, output_dir=folder, memory_bytes=256*1024**2, heartbeat_sec=0)
        self.assertEqual(result['status'], 'cleanup_failed')
        self.assertFalse(result['all_processes_reaped'])
        self.assertEqual(result['storage_status'], 'deferred_unconfirmed_process_exit')
        self.assertEqual((folder/'stdout.log').read_bytes(), b'x'*4096)
        self.assertFalse((folder/'stdout.log.gz').exists())
        saved = storage.read_json(folder/'process_result.json')
        self.assertEqual(saved['status'], 'cleanup_failed')
        self.assertEqual(saved['storage_status'], 'deferred_unconfirmed_process_exit')

    def test_supervisor_finalizes_only_after_confirmed_exit(self):
        storage = self.storage()
        import supervised
        folder = self.output/'confirmed'
        with patch.object(storage, 'AUTO_COMPRESS_MIN_BYTES', 1024), patch.dict(os.environ):
            for key in ('VRQO_PROGRESS_FILE', 'VRQO_PARENT_STATUS_FILE', 'VRQO_DEADLINE_MONOTONIC'):
                os.environ.pop(key, None)
            result = supervised.run_limited([sys.executable, '-B', '-c', 'import sys;sys.stdout.write("x"*4096)'],
                cwd=self.root, output_dir=folder, memory_bytes=256*1024**2, heartbeat_sec=0)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['all_processes_reaped'])
        self.assertEqual(result['storage_status'], 'complete')
        self.assertFalse((folder/'stdout.log').exists())
        with storage.open_result(folder/'stdout.log', 'rb') as handle:
            self.assertEqual(handle.read(), b'x'*4096)

    def test_inner_ae_worker_preserves_inherited_blas_and_separate_aer_threads(self):
        self.storage()
        import ae_runtime
        import supervised
        original_supervisor = supervised.run_limited
        for index, (aer, inherited, expected) in enumerate(((2, '1', '1'), (1, '2', '2'), (2, None, '1'))):
            with self.subTest(aer=aer, inherited=inherited):
                folder = self.output/('ae_threads_'+str(index))
                def environment_worker(command, **kwargs):
                    # Keep the actual OS process/environment boundary; replace
                    # only the scientific worker with a small stdlib reporter.
                    code = ('import os,json,pathlib;pathlib.Path('+repr(str(folder/'worker_result.json'))+
                            ').write_text(json.dumps({"value":0.25,"meta":{"status":"completed",'
                            '"worker_threads":{k:os.environ.get(k) for k in '
                            '("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VRQO_BLAS_THREADS")}}}),encoding="utf-8")')
                    return original_supervisor([sys.executable, '-B', '-c', code], **kwargs)
                with patch.dict(os.environ), patch.object(supervised, 'run_limited', side_effect=environment_worker):
                    for key in ('VRQO_PROGRESS_FILE', 'VRQO_PARENT_STATUS_FILE', 'VRQO_DEADLINE_MONOTONIC', 'VRQO_BLAS_THREADS'):
                        os.environ.pop(key, None)
                    if inherited is not None:os.environ['VRQO_BLAS_THREADS'] = inherited
                    result = ae_runtime.run_ae_worker('fixed', {'seed': 7, 'backend_options': {'max_parallel_threads': aer}},
                        output_dir=folder, memory_bytes=256*1024**2)
                actual = result['meta']['worker_threads']
                self.assertEqual(actual['OMP_NUM_THREADS'], str(aer))
                for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VRQO_BLAS_THREADS'):
                    self.assertEqual(actual[key], expected, key)

    def test_invalid_inherited_ae_blas_team_fails_before_creating_output(self):
        self.storage()
        import ae_runtime
        def no_science(command, **kwargs):
            folder = Path(kwargs['output_dir']).parent
            runtime_identity.write_json_new(folder/'worker_result.json', {'value': .25, 'meta': {'status': 'completed'}})
            return {'status': 'passed', 'all_processes_reaped': True}
        for index, value in enumerate(('0', '-1', 'bad', '1.5', 'True')):
            folder = self.output/('invalid_blas_'+str(index))
            with self.subTest(value=value), patch.dict(os.environ, {'VRQO_BLAS_THREADS': value}), patch(
                    'supervised.run_limited', side_effect=no_science):
                with self.assertRaises(ValueError):
                    ae_runtime.run_ae_worker('fixed', {'seed': 7}, output_dir=folder)
                self.assertFalse(folder.exists())


if __name__ == '__main__':
    unittest.main()
