"""Full-sample Sharpe checks against the reference evalPerf convention."""
import unittest

import numpy as np
import pandas as pd

from backtest_overfitting.performance_metrics import pnl_sharpe_ratio
from backtest_overfitting.main import measure_performance


class PnLPerformanceTests(unittest.TestCase):
    def test_recorded_zero_population_std_and_inclusive_calendar_duration(self):
        # Four records: mean 0.5, population variance 1.25; window is 11 days.
        pnl = pd.Series([np.nan, 0., 1., -1., 2., np.nan],
                        index=pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06",
                                              "2020-01-07", "2020-01-09", "2020-01-11"]))
        expected = .5 / np.sqrt(1.25) * np.sqrt(4 * 365.25 / 11)
        result = pnl_sharpe_ratio(pnl, pnl.index[0], pnl.index[-1])
        self.assertAlmostEqual(result, expected)
        # Neither padding inactive days with zero nor removing recorded zeros is valid.
        self.assertNotAlmostEqual(result, pnl_sharpe_ratio(pnl.fillna(0.), pnl.index[0], pnl.index[-1]))
        self.assertNotAlmostEqual(result, pnl_sharpe_ratio(pnl[pnl != 0], pnl.index[0], pnl.index[-1]))


    def test_undefined_scores_and_invalid_values_are_explicit(self):
        for values in ([], [np.nan], [0., 1.], [np.nan, 0., 1.], [0., 0., 0.], [2., 2., 2.]):
            with self.subTest(values=values):
                self.assertTrue(np.isnan(pnl_sharpe_ratio(values, "2020-01-01", "2020-01-10")))
        for values in ([np.inf], [-np.inf], [0., 1., np.inf]):
            with self.assertRaises(ValueError):
                pnl_sharpe_ratio(values, "2020-01-01", "2020-01-10")
        for start, end in ((pd.NaT, "2020-01-10"), ("2020-01-01", pd.NaT),
                           ("2020-01-10", "2020-01-01")):
            with self.assertRaises(ValueError):
                pnl_sharpe_ratio([0., 1., 2.], start, end)

    def test_batch_and_main_use_one_full_window_for_every_strategy(self):
        dates = pd.date_range("2020-01-01", periods=10)
        pnl = pd.DataFrame({7: [np.nan, 0., 1., -1., 2.] + [np.nan] * 5,
                            42: [np.nan] * 4 + [0., 2., -2., 4.] + [np.nan] * 2}, index=dates)
        configurations = pd.DataFrame({"strategy_id": [7, 42], "entry_day": [3, 4]})
        result = measure_performance(pnl, configurations)
        expected = .5 / np.sqrt(1.25) * np.sqrt(4 * 365.25 / 10)
        np.testing.assert_allclose(result.sharpe, [expected, expected])
        self.assertEqual(result.strategy_id.tolist(), [7, 42])
        self.assertEqual(result.entry_day.tolist(), [3, 4])
        self.assertEqual(result.columns.tolist(), ["strategy_id", "sharpe", "entry_day"])


if __name__ == "__main__":
    unittest.main()
