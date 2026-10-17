import numpy as np
import pandas as pd

def random_walk(n_days=1000, initial_price=100.0, sigma=0.01, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    return prices


def random_walk_seasonal(n_days=1000, initial_price=100.0, sigma=0.01, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, n_days)

    # business-day dates
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)

    # trading-day number within each month
    trading_day = pd.Series(dates).groupby(dates.to_period("M")).cumcount() + 1

    # seasonal effect to first 5 trading days
    seasonal_mask = (trading_day <= 5).to_numpy()
    returns[seasonal_mask] += 0.25 * sigma

    # Construct prices
    prices = initial_price * np.exp(np.cumsum(returns))

    return prices