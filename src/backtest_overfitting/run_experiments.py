"""Run one paired experiment and its plots using a YAML configuration."""
import argparse

from .config import load_config
from .main import main as run_main
from .paths import ROOT
from .plot import plot_experiment


def main(config_path=ROOT / "configs" / "config.yaml"):
    config = load_config(config_path)
    run_main(experiment_name=config["experiment"], base_seed=config["seed"],
             S=config["blocks"], n_days=config["n_days"], n_times=1)
    plot_experiment(config["experiment"], config["seed"])


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs" / "config.yaml")
    main(parser.parse_args().config)


if __name__ == "__main__":
    cli()
