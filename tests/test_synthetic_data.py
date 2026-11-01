"""Checks for the additive price process and its caller."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest_overfitting.synthetic_data import random_walk, random_walk_seasonal, simulation_dates


class SyntheticDataTests(unittest.TestCase):
    def test_additive_walk_can_cross_zero_and_offset_only_translates_prices(self):
        with patch("backtest_overfitting.synthetic_data.Random") as generator:
            generator.return_value.gauss.side_effect = [1., -3., 1.] * 2
            np.testing.assert_array_equal(random_walk(n_days=3), [1., -2., -1.])
            generator.return_value.gauss.assert_called_with(0, 1.)
            np.testing.assert_array_equal(
                random_walk(n_days=3, initial_price=100.), [101., 98., 99.])

    def test_seasonal_shift_restarts_each_month_and_scales_with_sigma(self):
        # Literal source calendar and numBusinessDays convention, January 2000.
        for sigma in (1., 2.):
            ordinary = random_walk(n_days=30, sigma=sigma, seed=7)
            seasonal = random_walk_seasonal(n_days=30, sigma=sigma, seed=7)
            expected = np.zeros(30)
            expected[:4] = expected[17:22] = .25 * sigma
            np.testing.assert_allclose(
                np.diff(seasonal - ordinary, prepend=0.), expected, atol=1e-13)
        self.assertEqual(len(random_walk()), 1000)
        self.assertEqual(len(random_walk_seasonal()), 1000)


    def test_literal_source_calendar_and_seeded_gaussian_stream(self):
        from random import Random
        import datetime as dt
        dates = simulation_dates(1000)
        self.assertEqual(dates[0], pd.Timestamp("2000-01-03"))
        self.assertEqual(set(dates.dayofweek), {0, 1, 2, 3})
        rng = Random(19)
        increments = []
        for date in dates.date:
            value = rng.gauss(0, 1)
            cursor, number = date.replace(day=1), 1
            while True:
                cursor += dt.timedelta(days=1)
                if cursor >= date:
                    break
                if cursor.isoweekday() < 6:
                    number += 1
            increments.append(value + (.25 if number <= 5 else 0.))
        np.testing.assert_allclose(random_walk_seasonal(seed=19), np.cumsum(increments), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
