
import numpy as np
import pandas as pd

from backtest_overfitting import paths
from ras.core import ras_sharpe


def prepare_pbo_data_for_ras(returns):
    """
    Convert the saved PBO strategy P&L matrix into
    volatility-normalized observations for RAS.

    Exploratory convention:
    - Unrecorded P&L is treated as zero.
    - Volatility is calculated over the full calendar.
    - The original return matrix is not modified.
    """
    calendar_pnl = returns.fillna(0.0)

    volatility = calendar_pnl.std(axis=0, ddof=1)

    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("Invalid strategy volatility")

    X = calendar_pnl.div(volatility, axis=1).to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("RAS input contains non-finite values")

    return X


def run_ras_on_pbo_data(
    experiment="demo_experiment",
    seed=42,
    B=1000,
    delta=0.1,
    n_strategies=None,
):
    """
    Run the existing RAS implementation on saved
    PBO simulation data.
    """
    returns_dir = paths.processed_dir(experiment, seed) / "returns"

    results = {}

    for scenario in ("no_signal", "seasonal"):
        returns = pd.read_parquet(
            returns_dir / f"{scenario}.parquet"
        )

        if n_strategies is not None:
            returns = returns.iloc[:, :n_strategies]

        X = prepare_pbo_data_for_ras(returns)

        # Call other RAS function
        ras_result = ras_sharpe(
            X,
            delta=delta,
            B=B,
            seed=seed,
        )

        results[scenario] = {
            "strategies": X.shape[1],
            "observations": X.shape[0],
            "complexity": ras_result["rademacher_complexity"],
            "haircut": ras_result["haircut"],
            "best_empirical_sharpe": float(
                np.max(ras_result["empirical_sharpe"])
            ),
            "best_adjusted_bound": float(
                np.max(ras_result["lower_bound"])
            ),
            "positive_bounds": int(
                np.sum(ras_result["lower_bound"] > 0)
            ),
        }

    return pd.DataFrame.from_dict(results, orient="index")

if __name__ == "__main__":
    results = run_ras_on_pbo_data(
        experiment="demo_experiment",
        seed=42,
        B=1000,
        delta=0.1,
    )

    print("\nRAS Results Using PBO Simulated Data")
    print(results.to_string())
    
# To run this script, you can use the following code:
#
# from ras.pbo_comparison import run_ras_on_pbo_data
#
# results = run_ras_on_pbo_data(
#     experiment="demo_experiment",
#     seed=42,
#     B=1000,
#     delta=0.1,
# )
#
# print(results.to_string())
