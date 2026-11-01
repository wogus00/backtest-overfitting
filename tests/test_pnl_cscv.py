"""Recorded P&L, calendar exposure and fixed-universe CSCV regression checks."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from backtest_overfitting.cscv import cscv, generate_combinations, compute_relative_rank
from backtest_overfitting.performance_metrics import calendar_day_weights, column_pnl_sharpe_ratios, pnl_sharpe_ratio
from backtest_overfitting.main import run_cscv, main
from backtest_overfitting import plot
from backtest_overfitting import paths, run_experiments, analyze_performance


class PnLCSCVTests(unittest.TestCase):
    def setUp(self):
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stderr.__enter__()
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_calendar_exposure_partitions_weekends_without_counting_omitted_blocks(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-07", "2020-01-09"])
        weights = calendar_day_weights(dates)
        np.testing.assert_array_equal(weights, [4., 1., 2., 1.])
        self.assertEqual(weights.sum(), 8)
        self.assertEqual(weights[[0, 2]].sum(), 6)
        for bad in (dates[::-1], dates[[0, 0]], pd.DatetimeIndex([]), pd.DatetimeIndex([pd.NaT])):
            with self.assertRaises(ValueError):
                calendar_day_weights(bad)

    def test_vectorized_scores_match_full_sample_scalar_and_keep_recorded_zeros(self):
        dates = pd.bdate_range("2020-01-01", periods=24)
        values = np.random.default_rng(11).normal(size=(24, 5))
        values[::3, 0] = np.nan
        values[1::3, 1] = 0.
        values[:, 2] = values[:, 0]
        values[:, 3] = np.nan
        values[:, 4] = 1.
        expected = [pnl_sharpe_ratio(values[:, i], dates[0], dates[-1]) for i in range(5)]
        actual = column_pnl_sharpe_ratios(values, calendar_day_weights(dates).sum())
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(actual[0], actual[2])
        for duration in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                column_pnl_sharpe_ratios(values, duration)

    def test_masked_splits_match_independent_scores_ranks_and_additive_totals(self):
        dates = pd.bdate_range("2020-01-01", periods=12)
        values = np.random.default_rng(12).normal(size=(12, 2))
        values[::3, 0] = np.nan
        values[::4, 1] = 0.
        matrix = pd.DataFrame(values, index=dates, columns=[10, 42])
        result = run_cscv(matrix, S=4)
        weights = calendar_day_weights(dates)
        blocks = np.split(np.arange(12), 4)
        for row, (ins, oos) in zip(result.itertuples(), generate_combinations(4)):
            i, o = np.concatenate([blocks[k] for k in ins]), np.concatenate([blocks[k] for k in oos])
            def scores(indices):
                output = []
                for col in range(2):
                    recorded = values[indices, col]
                    recorded = recorded[~np.isnan(recorded)]
                    output.append(recorded.mean() / recorded.std(ddof=0) * np.sqrt(len(recorded) * 365.25 / weights[indices].sum()))
                return np.array(output)
            is_scores, oos_scores = scores(i), scores(o)
            winner = np.argmax(is_scores)
            self.assertEqual(row.selected_idx, winner)
            self.assertEqual(row.strategy_id, matrix.columns[winner])
            self.assertAlmostEqual(row.performance_is, is_scores[winner])
            self.assertAlmostEqual(row.performance_oos, oos_scores[winner])
            relative_rank = rankdata(oos_scores)[winner] / 3
            self.assertAlmostEqual(row.relative_rank, relative_rank)
            self.assertAlmostEqual(row.logit, np.log(relative_rank / (1 - relative_rank)))
            self.assertAlmostEqual(row.benchmark_oos, oos_scores.mean())
            self.assertAlmostEqual(row.oos_total_pnl, np.nansum(values[o, winner]))
            self.assertEqual(row.is_calendar_days + row.oos_calendar_days, weights.sum())
        self.assertNotIn("oos_total_return", result)
        self.assertEqual(compute_relative_rank([3., 1., 1.], 1), (1.5, .375))

    def test_undefined_candidates_are_excluded_once_not_per_split(self):
        dates = pd.bdate_range("2020-01-01", periods=12)
        values = np.random.default_rng(5).normal(size=(12, 4))
        values[:6, 0] = np.nan  # A half-sample has no records.
        values[:6, 2] = 7.      # A half-sample is constant.
        result = run_cscv(pd.DataFrame(values, index=dates, columns=[9, 10, 11, 12]), S=4)
        eligibility = result.attrs["eligibility"]
        self.assertEqual(eligibility.eligible.tolist(), [False, True, False, True])
        self.assertTrue(result.strategy_id.isin([10, 12]).all())
        self.assertTrue(result.selected_idx.isin([1, 3]).all())
        self.assertTrue(result.n_strategies.eq(2).all())
        with self.assertRaisesRegex(ValueError, "Fewer than two"):
            run_cscv(pd.DataFrame(values[:, :3], index=dates), S=4)
        with self.assertRaises(ValueError):
            cscv(values, S=4, calendar_days=np.ones(11))
        values[0, 1] = np.inf
        with self.assertRaises(ValueError):
            cscv(values, S=4, calendar_days=np.ones(12))


    def test_pipeline_saves_eligibility_dates_pnl_and_loadable_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("backtest_overfitting.paths.DATA_DIR", root / "data"), \
                    patch("backtest_overfitting.paths.RESULT_DIR", root / "results/replication"):
                with self.assertRaisesRegex(ValueError, "divisible"):
                    main("invalid", S=16, n_days=1000)
                self.assertFalse((root / "data").exists())
                config = root / "smoke.yaml"
                config.write_text("experiment: smoke\nseed: 1\nblocks: 4\nn_days: 80\n")
                run_experiments.main(config)
                self.assertTrue((paths.processed_dir("smoke", 1) / "strategy_configurations.parquet").is_file())
                self.assertTrue((paths.figures_dir("smoke", 1) / "comparison.pdf").is_file())
                self.assertTrue((paths.tables_dir("smoke", 1) / "pooled_dominance_summary.csv").is_file())
                self.assertFalse((paths.result_dir("smoke", 1) / "returns").exists())
                self.assertTrue(all(file.suffix in (".png", ".pdf")
                                    for file in paths.figures_dir("smoke", 1).iterdir()))
                with patch("sys.argv", ["analyze_performance", "--experiment", "smoke", "--seed", "1"]):
                    analyze_performance.main()
                self.assertTrue((paths.tables_dir("smoke", 1) / "seasonal/sharpe/performance.csv").is_file())
                with self.assertRaises(FileExistsError):
                    main("smoke", S=4, n_days=80, base_seed=1)
                for name in ("no_signal", "seasonal"):
                    data = plot.load_dataset("smoke", 1, name)
                    self.assertTrue(data["is_pnl"])
                    self.assertEqual(data["T"], 80)
                    self.assertEqual(data["S"], 4)
                    self.assertEqual(set(data["returns"].index.dayofweek), {0, 1, 2, 3})
                    self.assertIn("oos_total_pnl", data["cscv"])

    def test_existing_raw_processed_or_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(paths, "DATA_DIR", root / "data"), \
                    patch.object(paths, "RESULT_DIR", root / "results/replication"):
                for index, location in enumerate((paths.data_dir, paths.processed_dir, paths.result_dir)):
                    name = f"existing_{index}"
                    occupied = location(name, 1)
                    occupied.mkdir(parents=True)
                    marker = occupied / "preserve.txt"
                    marker.write_text("existing output")
                    with self.assertRaises(FileExistsError):
                        main(name, S=4, n_days=80)
                    self.assertEqual(marker.read_text(), "existing output")
                    self.assertEqual(list(occupied.iterdir()), [marker])


    def test_pooled_replay_uses_every_strategy_and_checks_saved_scores(self):
        from backtest_overfitting.main import run_cscv
        from backtest_overfitting.performance_metrics import calendar_day_weights, column_pnl_sharpe_ratios, sharpe_ratio, column_sharpe_ratios
        matrix = pd.DataFrame(np.random.default_rng(8).normal(size=(12, 3)),
                              index=pd.bdate_range("2000-01-03", periods=12), columns=[10, 20, 30])
        for is_pnl in (False, True):
            sample = matrix.copy()
            if is_pnl:
                sample.iloc[::4, 0] = np.nan
            saved = run_cscv(sample, S=2, metric=None if is_pnl else sharpe_ratio)
            curve, summary = plot.pooled_dominance(sample, saved, 2, is_pnl)
            weights = calendar_day_weights(sample.index)
            expected = []
            for indices in (np.arange(6, 12), np.arange(6)):
                values = sample.iloc[indices].to_numpy()
                expected.extend(column_pnl_sharpe_ratios(values, weights[indices].sum())
                                if is_pnl else column_sharpe_ratios(values))
            expected = np.array(expected)
            self.assertEqual(summary["selected_count"], 2)
            self.assertEqual(summary["benchmark_count"], 6)
            np.testing.assert_allclose(curve.cdf_pooled,
                (expected[:, None] <= curve.score.to_numpy()).mean(axis=0))
            x = curve.score.to_numpy()[:, None]
            expected_integral = np.maximum(x - expected, 0).mean(axis=1) - np.maximum(x - saved.performance_oos.to_numpy(), 0).mean(axis=1)
            np.testing.assert_allclose(curve.integrated_cdf_difference, expected_integral, atol=1e-12)
            saved.loc[0, "performance_oos"] += 1
            with self.assertRaisesRegex(ValueError, "disagree"):
                plot.pooled_dominance(sample, saved, 2, is_pnl)


if __name__ == "__main__":
    unittest.main()
