"""Trade-level checks for the CSCV_3.py conventions, independent of Sharpe."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from backtest_overfitting.strategies import (generate_strategy_configurations, generate_strategy_returns,
                        generate_strategy_returns_old, run_seasonal_strategy)


class SeasonalTradeTests(unittest.TestCase):
    def prices(self, values):
        return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))

    def test_entry_zero_exit_pnl_and_signed_price_changes(self):
        prices = self.prices([100., 90., 93., 91., 95., 100.])
        expected = np.array([np.nan, 0., 3., -2., np.nan, np.nan])
        for side in (-1, 1):
            actual = run_seasonal_strategy(prices, 3, 2, -10, side)
            np.testing.assert_allclose(actual, expected * side)
            shifted = run_seasonal_strategy(prices - 1000, 3, 2, -10, side)
            pd.testing.assert_series_equal(actual, shifted)

    def test_stop_uses_previous_pnl_and_can_reenter_after_profitable_exit(self):
        prices = self.prices([0., 0., -2., 5., 9., 10.])
        # The loss at index 2 is checked at index 3, which still earns exit P&L.
        # That gain permits re-entry (recorded zero) at index 4 in the source.
        expected = [np.nan, 0., -2., 7., 0., 1.]
        np.testing.assert_allclose(run_seasonal_strategy(prices, 3, 4, -1, 1), expected)
        # Equality also closes the position; a wider fixed-sigma stop does not.
        np.testing.assert_allclose(run_seasonal_strategy(prices, 3, 4, -1, 1, sigma=2), expected)
        np.testing.assert_allclose(run_seasonal_strategy(prices, 3, 4, -1, 1, sigma=3),
                                   [np.nan, 0., -2., 7., 4., 1.])

    def test_first_month_offset_and_monthly_reset_with_open_position(self):
        prices = self.prices(np.arange(65, dtype=float))
        actual = run_seasonal_strategy(prices, 1, 20, -10, 1)
        # Source first-month numbering starts at 2, so there is no January day 1.
        self.assertTrue(actual.loc["2020-01"].isna().all())
        self.assertEqual(actual.loc["2020-02-03"], 0.)
        self.assertEqual(actual.loc["2020-03-02"], 1.)
        self.assertEqual(actual.loc["2020-03-03"], 1.)
        # A reference date at the first price row is skipped by the source loop.
        first_row_entry = run_seasonal_strategy(prices, 2, 4, -10, 1)
        self.assertTrue(first_row_entry.loc["2020-01"].isna().all())
        self.assertEqual(first_row_entry.loc["2020-02-04"], 0.)

    def test_candidate_grid_and_insufficient_observation_filter(self):
        grid = generate_strategy_configurations()
        self.assertEqual(len(grid), 29 * 21 * 11 * 2)
        self.assertEqual(set(grid.holding_period), set(range(21)))
        self.assertEqual(set(grid.stop_loss), set(range(-10, 1)))
        prices = self.prices([100., 90., 93., 91., 95., 100.])
        configs = pd.DataFrame([
            (10, 3, 2, -10, 1),   # entry zero and two earned P&Ls: retained
            (11, 3, 1, -10, 1),   # only two observations: omitted
            (12, 3, 0, -10, 1),   # zero holding period: no trade
            (13, 3, 2, 0, 1),     # zero threshold: strict inequality prevents entry
            (14, 29, 2, -10, 1),  # reference ordinal absent
        ], columns=grid.columns)
        pnl = generate_strategy_returns(prices, configs)
        self.assertEqual(pnl.columns.tolist(), [10])
        self.assertTrue(pnl.index.equals(prices.index))
        np.testing.assert_array_equal(pnl[10].notna(), [False, True, True, True, False, False])
        np.testing.assert_allclose(pnl[10], [np.nan, 0., 3., -2., np.nan, np.nan])
        pd.testing.assert_frame_equal(pnl, generate_strategy_returns_old(prices, configs))
        observed_grid = generate_strategy_configurations(prices.index)
        self.assertEqual(set(observed_grid.entry_day), set(range(2, 8)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pnl.parquet"
            pnl.to_parquet(path)
            pd.testing.assert_frame_equal(pd.read_parquet(path), pnl, check_freq=False)


    def test_invalid_prices_rules_and_duplicate_ids_are_rejected(self):
        prices = self.prices([1., 2., 3., 4.])
        for invalid in (prices.iloc[::-1], prices.iloc[[0, 0]], prices * np.nan, prices.iloc[:0]):
            with self.assertRaises(ValueError):
                run_seasonal_strategy(invalid, 3, 2, -1, 1)
        for kwargs in ({"entry_day": 0}, {"holding_period": 1.5}, {"stop_loss": 1},
                       {"side": 0}, {"sigma": 0}, {"sigma": np.nan}):
            parameters = dict(entry_day=3, holding_period=2, stop_loss=-1, side=1)
            parameters.update(kwargs)
            with self.assertRaises(ValueError):
                run_seasonal_strategy(prices, **parameters)
        config = generate_strategy_configurations().iloc[[0, 0]]
        with self.assertRaisesRegex(ValueError, "unique"):
            generate_strategy_returns(prices, config)


if __name__ == "__main__":
    unittest.main()
