from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from backtest_overfitting import run_experiments


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.filename = Path(self.directory.name) / "config.yaml"
        self.config = dict(experiment="configured_run", seed=7, blocks=4, n_days=80)

    def write_config(self, config):
        self.filename.write_text(yaml.safe_dump(config))

    def test_runner_passes_config_and_plots_only_after_success(self):
        self.write_config(self.config)
        for error in (None, RuntimeError("failed experiment")):
            with self.subTest(error=error), \
                    patch.object(run_experiments, "run_main", side_effect=error) as run, \
                    patch.object(run_experiments, "plot_experiment") as plot:
                if error:
                    with self.assertRaisesRegex(RuntimeError, "failed experiment"):
                        run_experiments.main(self.filename)
                    plot.assert_not_called()
                else:
                    run_experiments.main(self.filename)
                    plot.assert_called_once_with("configured_run", 7)
                run.assert_called_once_with(experiment_name="configured_run", base_seed=7,
                                            S=4, n_days=80, n_times=1)

    def test_invalid_settings_fail_before_running(self):
        invalid = [None, [], {}, {**self.config, "typo": 1}]
        for key, values in {
            "experiment": ["", "../escape", "/absolute", 42],
            "seed": [True, "7", 1.5],
            "blocks": [True, 0, 3, "4"],
            "n_days": [False, 0, 2, 81, "80"],
        }.items():
            invalid.extend({**self.config, key: value} for value in values)
        with patch.object(run_experiments, "run_main") as run:
            for config in invalid:
                with self.subTest(config=config):
                    self.write_config(config)
                    with self.assertRaises(ValueError):
                        run_experiments.main(self.filename)
            run.assert_not_called()

