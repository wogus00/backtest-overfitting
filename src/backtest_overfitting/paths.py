"""Repository paths for the editable research project."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results" / "replication"


def data_dir(experiment, seed):
    return DATA_DIR / "raw" / experiment / f"seed_{seed}"


def processed_dir(experiment, seed):
    return DATA_DIR / "processed" / experiment / f"seed_{seed}"


def result_dir(experiment, seed):
    return RESULT_DIR / experiment / f"seed_{seed}"


def tables_dir(experiment, seed):
    return result_dir(experiment, seed) / "tables"


def figures_dir(experiment, seed):
    return result_dir(experiment, seed) / "figures"


def legacy_directories():
    """Locate the old unseeded layout used only by plotting.py."""
    for root in (ROOT / "src", ROOT):
        results = root / "results"
        if all((results / name / "sharpe" / "cscv.parquet").is_file()
               for name in ("no_signal", "seasonal")):
            return root / "data" / "prices" / "synthetic", results
    raise FileNotFoundError(
        "Legacy experiment files were not found under src/results or results. "
        "For current experiments, use python -m backtest_overfitting.plot."
    )
