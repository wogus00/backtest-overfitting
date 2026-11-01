"""Additive walks using the published CSCV_3.py calendar and Gaussian RNG."""
from random import Random

import numpy as np
import pandas as pd


def simulation_dates(n_days=1000):
    """Literal source calendar: isoweekday() < 5 excludes Friday."""
    if isinstance(n_days, (bool, np.bool_)) or not isinstance(n_days, (int, np.integer)) or n_days < 1:
        raise ValueError("n_days must be a positive integer.")
    return pd.date_range("2000-01-01", periods=n_days,
                         freq=pd.offsets.CustomBusinessDay(weekmask="Mon Tue Wed Thu"))


def random_walk(n_days=1000, initial_price=0.0, sigma=1.0, seed=42):
    """Additive Gaussian price walk; sigma is in price units, not percent."""
    simulation_dates(n_days)
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(initial_price):
        raise ValueError("sigma must be positive and finite; initial_price must be finite.")
    rng = Random(seed)
    increments = np.array([rng.gauss(0, sigma) for _ in range(n_days)])
    return initial_price + np.cumsum(increments)


def random_walk_seasonal(n_days=1000, initial_price=0.0, sigma=1.0, seed=42):
    """Use the source's numBusinessDays <= 5 seasonal shift, including quirks."""
    prices = random_walk(n_days, initial_price, sigma, seed)
    dates = simulation_dates(n_days)
    starts = dates.to_numpy(dtype="datetime64[M]").astype("datetime64[D]")
    days = dates.to_numpy(dtype="datetime64[D]")
    # Source counts weekdays strictly between month-start and date, plus one.
    business_day = 1 + np.maximum(0, np.busday_count(starts + 1, days))
    shift = np.where(business_day <= 5, .25 * sigma, 0.)
    return prices + np.cumsum(shift)
