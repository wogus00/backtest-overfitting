"""Checks against Sections 3-4 equations, using hand-computable examples."""
import unittest

import numpy as np
import pandas as pd

from backtest_overfitting.overfit_statistics import performance_degradation, probability_of_loss, stochastic_dominance
from backtest_overfitting.cscv_validation import (
    validate_effect_of_S, validate_effect_of_N, validate_temporal_dependence,
)
from backtest_overfitting.main import compute_statistics, run_validation


class PaperStatisticsTests(unittest.TestCase):
    def test_pooled_dominance_normalizes_unequal_populations_and_integrates_steps(self):
        selected = pd.DataFrame({"performance_oos": [0., 2.]})
        result = stochastic_dominance(selected, benchmark_scores=[-1., 0., 0., 3.])
        np.testing.assert_array_equal(result["x"], [-1., 0., 2., 3.])
        np.testing.assert_array_equal(result["cdf_selected"], [0., .5, 1., 1.])
        np.testing.assert_array_equal(result["cdf_benchmark"], [.25, .75, .75, 1.])
        np.testing.assert_allclose(result["sd2"], [0., .25, .75, .5])
        self.assertFalse(result["first_order_dominance"])
        self.assertTrue(result["second_order_dominance"])
        for benchmark in ([], [np.nan], [np.inf], [[0., 1.]]):
            with self.assertRaises(ValueError):
                stochastic_dominance(selected, benchmark_scores=benchmark)

    def test_degradation_regresses_oos_on_is_with_intercept(self):
        frame = pd.DataFrame({'performance_is': [1., 2., 3.],
                              'performance_oos': [3., 1., -1.]})
        result = performance_degradation(frame)
        np.testing.assert_allclose([result['slope'], result['intercept'], result['R^2']], [-2., 5., 1.])
        self.assertTrue(result['defined'])

    def test_undefined_regression_does_not_block_other_statistics(self):
        frame = pd.DataFrame({'performance_is': [1., 1., 1.],
                              'performance_oos': [-1., 0., 1.],
                              'benchmark_oos': [0., 0., 0.],
                              'logit': [-1., 0., 1.]})
        summary, _ = compute_statistics(frame, 'sharpe', benchmark=0.)
        self.assertFalse(summary['degradation_defined'])
        self.assertTrue(np.isnan(summary['degradation_slope']))
        self.assertEqual(summary['pbo'], 1/3)
        self.assertEqual(summary['probability_of_loss'], 1/3)
        self.assertNotIn('probability_of_compounded_loss', summary)
        self.assertEqual(summary['probability_below_oos_benchmark'], 1/3)
        one = performance_degradation(frame.iloc[:1])
        self.assertFalse(one['defined'])
        frame['performance_is'] = [1., 2., 3.]
        frame['performance_oos'] = 2.
        constant_y = performance_degradation(frame)
        self.assertEqual(constant_y['slope'], 0.)
        self.assertEqual(constant_y['intercept'], 2.)
        self.assertTrue(np.isnan(constant_y['R^2']))

    def test_loss_uses_score_not_compounded_return_or_custom_benchmark(self):
        frame = pd.DataFrame({'performance_is': [1., 2., 3.],
                              'performance_oos': [-1., 0., 1.],
                              'benchmark_oos': [0., 0., 0.],
                              'logit': [-1., 0., 1.],
                              'oos_total_return': [.2, -.1, -.1]})
        self.assertEqual(probability_of_loss(frame), 1/3)
        summary, _ = compute_statistics(frame, 'sharpe', benchmark=2.)
        self.assertEqual(summary['probability_of_loss'], 1/3)
        self.assertEqual(summary['probability_below_oos_benchmark'], 1.)
        # +50%, -40%: positive mean/Sharpe, negative compounded return.
        frame = pd.DataFrame({'performance_oos': [.05 / np.std([.5, -.4], ddof=1)],
                              'oos_total_return': [1.5 * .6 - 1]})
        self.assertEqual(probability_of_loss(frame), 0.)

    def test_dominance_direction_strictness_and_step_integral(self):
        cases = [([1., 2.], [0., 1.], True, True),
                 ([1., 1.], [0., 2.], False, True),
                 ([0., 2.], [1., 1.], False, False),
                 ([0., 1.], [0., 1.], False, False),
                 ([0., 3.], [1., 2.], False, False)]
        for selected, benchmark, fsd, ssd in cases:
            with self.subTest(selected=selected, benchmark=benchmark):
                result = stochastic_dominance(pd.DataFrame({
                    'performance_oos': selected, 'benchmark_oos': benchmark}))
                self.assertEqual(result['first_order_dominance'], fsd)
                self.assertEqual(result['second_order_dominance'], ssd)
                x = result['x'][:, None]
                # Integral of an empirical CDF equals mean positive shortfall.
                expected = np.maximum(x - benchmark, 0).mean(axis=1) - np.maximum(x - selected, 0).mean(axis=1)
                np.testing.assert_allclose(result['sd2'], expected, atol=1e-12)

    def test_all_statistics_reject_missing_or_nonfinite_observations(self):
        for values in ([], [np.nan], [np.inf], [-np.inf], [-1., np.nan]):
            frame = pd.DataFrame({name: values for name in
                                  ['performance_is', 'performance_oos', 'benchmark_oos']})
            for function in (performance_degradation, probability_of_loss, stochastic_dominance):
                with self.subTest(function=function.__name__, values=values), self.assertRaises(ValueError):
                    function(frame)
        frame = pd.DataFrame({'performance_oos': [1., 2.], 'benchmark_oos': [0., np.nan]})
        with self.assertRaises(ValueError):
            stochastic_dominance(frame)


class PaperDiagnosticsTests(unittest.TestCase):
    def test_s_counts_sample_sizes_and_paper_heuristic(self):
        result = validate_effect_of_S(pd.DataFrame(np.zeros((1008, 2))), [4, 8, 16])
        self.assertEqual(result.n_combinations.tolist(), [6, 70, 12870])
        self.assertEqual(result.min_block_size.tolist(), [252, 126, 63])
        self.assertTrue(result.is_observations.eq(504).all())
        self.assertTrue(result.oos_observations.eq(504).all())
        np.testing.assert_allclose(result.paper_binomial_se_upper_bound, np.sqrt(.25 / np.array([6, 70, 12870])))
        self.assertNotIn('pbo', result)

    def test_n_diagnostics_are_theoretical_and_defaults_fit_matrix(self):
        frame = pd.DataFrame(np.zeros((12, 4)))
        result = validate_effect_of_N(frame, [2, 4])
        np.testing.assert_allclose(result.rank_resolution, [1/3, 1/5])
        np.testing.assert_allclose(result.min_logit, -np.log([2, 4]))
        np.testing.assert_allclose(result.max_logit, np.log([2, 4]))
        self.assertEqual(result.unique_logit_values.tolist(), [2, 4])
        self.assertEqual(validate_effect_of_N(frame).N.tolist(), [4])
        for n in (1, 5, 2.5, True):
            with self.subTest(N=n), self.assertRaises(ValueError):
                validate_effect_of_N(frame, [n])

    def test_chronology_symmetry_and_main_paper_scope(self):
        frame = pd.DataFrame(np.zeros((12, 4)))
        result = validate_temporal_dependence(frame, S=4)
        self.assertEqual(len(result), 6)
        self.assertTrue(result[['is_chronological', 'oos_chronological', 'no_overlap',
                                'complete_split', 'equal_sample_size', 'oos_reused_as_is']].all().all())
        self.assertGreater(result.is_boundary_gaps.max(), 0)
        diagnostics = run_validation(frame, [2, 4], [2, 4], S=4)
        self.assertEqual(set(diagnostics), {'effect_of_S', 'effect_of_N', 'temporal_dependence'})
        with self.assertRaises(ValueError):
            run_validation(frame, [2, 4], [2, 4], S=4, estimated_pbo=[.2])


if __name__ == '__main__':
    unittest.main()
