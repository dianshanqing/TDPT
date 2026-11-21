"""Summary keeps science, with one authoritative copy of verbose MPS logs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validation
from runtime_identity import write_json_new


class MPSLogStorageTests(unittest.TestCase):
    def test_summary_references_exact_authoritative_log_without_mutating_row(self):
        self.assertTrue(callable(getattr(validation,'_probe_summary_row',None)))
        log='MPS gate 数据\n'*50000
        row={'status':'complete','probabilities':[0.,1e-10,1.-1e-10,0.],
             'reference':[0.,1e-10,1.-1e-10,0.],'rare_accuracy_status':'passed',
             'strict_status':'passed','metadata':{'method':'matrix_product_state',
                 'MPS_log_data':log,'matrix_product_state_max_bond_dimension':128}}
        with tempfile.TemporaryDirectory() as td,patch('runtime_identity.PROJECT_ROOT',Path(td)):
            path=Path(td)/'outputs'/'run'/'case.json';write_json_new(path,row)
            summary=validation._probe_summary_row(row,path)
            self.assertEqual(row['metadata']['MPS_log_data'],log)
            self.assertNotIn('MPS_log_data',summary['metadata'])
            self.assertLess(len(json.dumps(summary)),3000)
            for key in ('probabilities','reference','rare_accuracy_status','strict_status'):
                self.assertEqual(summary[key],row[key])
            self.assertEqual(summary['metadata']['matrix_product_state_max_bond_dimension'],128)
            ref=summary['metadata_log_references']['MPS_log_data']
            self.assertEqual(ref['path'],'case.json')
            self.assertEqual(ref['json_pointer'],'/metadata/MPS_log_data')
            self.assertEqual(ref['file_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(ref['file_bytes'],path.stat().st_size)
            self.assertEqual(ref['log_sha256'],hashlib.sha256(log.encode('utf-8')).hexdigest())
            self.assertEqual(ref['log_utf8_bytes'],len(log.encode('utf-8')))

    def test_no_log_and_failed_rows_keep_original_fields(self):
        self.assertTrue(callable(getattr(validation,'_probe_summary_row',None)))
        for row in ({'status':'failed','error':'backend failure'},
                    {'status':'complete','metadata':{'method':'matrix_product_state'}}):
            self.assertEqual(validation._probe_summary_row(row,Path('unused.json')),row)


if __name__=='__main__':unittest.main()
