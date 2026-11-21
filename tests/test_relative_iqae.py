"""Truth-independent relative stopping, confidence spending and budget failures."""
import inspect
import math
import unittest
from unittest.mock import patch
import numpy as np
from iqae import estimate_relative_iqae, relative_interval_stop
from stats_utils import sigma_report


class RelativeIQAE(unittest.TestCase):
    def test_midpoint_sufficient_bound(self):
        self.assertTrue(relative_interval_stop([.2,.204]))
        self.assertFalse(relative_interval_stop([0.,.001]))
        self.assertFalse(relative_interval_stop([.2,.205]))
        for ci in ([.3,.2],[-.1,.1],[0.,float('nan')],[0.,1.1]):
            with self.assertRaises(ValueError):relative_interval_stop(ci)

    def test_truth_not_in_estimator_interface(self):
        args=inspect.signature(estimate_relative_iqae).parameters
        self.assertNotIn('p',args);self.assertNotIn('reference',args)

    def test_realistic_ideal_count_stream_and_spending(self):
        rng=np.random.default_rng(123);calls=[];p=.2
        def sample(k,n):
            calls.append((k,n));return int(rng.binomial(n,math.sin((2*k+1)*math.asin(math.sqrt(p)))**2))
        r=estimate_relative_iqae(sample)
        self.assertEqual(r['status'],'completed')
        self.assertTrue(relative_interval_stop(r['confidence_interval']))
        self.assertAlmostEqual(r['estimate'],sum(r['confidence_interval'])/2)
        self.assertLessEqual(r['failure_probability_spent'],.05)
        self.assertFalse(r['full_pipeline_certified'])
        self.assertEqual(r['costs']['queries_A_and_inverse'],sum(n*(2*k+1) for k,n in calls))
        self.assertEqual(r['costs']['A_calls']+r['costs']['A_inverse_calls'],r['costs']['queries_A_and_inverse'])
        for i,row in enumerate(r['rounds'],1):
            self.assertAlmostEqual(row['alpha'],.05/(i*(i+1)))
            self.assertEqual(row['shots'],512) # no adaptive cumulative pooling

    def test_power_budget_exit_is_not_success(self):
        r=estimate_relative_iqae(lambda k,n:0,max_power=0)
        self.assertEqual(r['status'],'budget_limited');self.assertIsNone(r['estimate'])
        self.assertIsNone(r['confidence_interval']);self.assertFalse(r['precision_target_met'])
        self.assertEqual(r['costs']['max_power'],0)

    def test_shots_and_queries_cap_before_sampling(self):
        calls=[]
        def sample(k,n):calls.append(n);return 0
        r=estimate_relative_iqae(sample,max_shots=10)
        self.assertEqual(calls,[10]);self.assertEqual(r['costs']['shots'],10)
        self.assertEqual(r['status'],'budget_limited')
        calls.clear();r=estimate_relative_iqae(sample,max_queries=1)
        self.assertEqual(calls,[]);self.assertEqual(r['costs']['shots'],0)

    def test_bad_counts_and_exception_fail(self):
        for value in (None,0.1,-1,513,True):
            r=estimate_relative_iqae(lambda k,n:value)
            self.assertEqual(r['status'],'failed');self.assertFalse(r['precision_target_met'])

    def test_nonfinite_binomial_interval_is_explicit_failure(self):
        with patch('scipy.stats.beta.ppf',return_value=float('nan')):
            r=estimate_relative_iqae(lambda k,n:100)
        self.assertEqual(r['status'],'invalid_interval')
        self.assertFalse(r['precision_target_met']);self.assertIsNone(r['estimate'])

    def test_inconsistent_batches_do_not_select_a_result(self):
        it=iter([100,512,0,512,0,512])
        r=estimate_relative_iqae(lambda k,n:next(it))
        self.assertNotEqual(r['status'],'completed');self.assertIsNone(r['estimate'])

    def test_parameter_validation(self):
        for kwargs in ({'confidence':1},{'relative_error':0},{'max_power':-1},
                       {'max_shots':0},{'shots':True},{'max_queries':float('inf')}):
            with self.assertRaises(ValueError):estimate_relative_iqae(lambda k,n:0,**kwargs)

    def test_combined_standard_errors(self):
        r=sigma_report(.4,.3,.03,.04)
        self.assertAlmostEqual(r.z_sigma,2);self.assertTrue(r.pass_3sigma)
        self.assertTrue(math.isinf(sigma_report(.4,.3,0,0).z_sigma))
        self.assertFalse(sigma_report(.4,.3,0,0).pass_3sigma)
        self.assertEqual(sigma_report(.3,.3,0,0).z_sigma,0)
        with self.assertRaises(ValueError):sigma_report(.3,.3,-1)


if __name__=='__main__':unittest.main()
