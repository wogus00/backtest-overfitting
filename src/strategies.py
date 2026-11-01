import itertools
import numpy as np
import pandas as pd


def _validate_prices(prices):
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a Pandas Series.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")
    if (prices.empty or prices.index.hasnans or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing):
        raise ValueError("prices must have nonempty, unique, increasing dates.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prices must be finite.")
    return values


def _entry_positions(dates):
    months = dates.to_period("M")
    ordinals = pd.Series(1, index=dates).groupby(months).cumsum().to_numpy()
    first = dates[0].normalize().replace(day=1)
    interior = pd.date_range(first + pd.Timedelta(days=1),
                             dates[0].normalize() - pd.Timedelta(days=1))
    # Preserve getRefDates_MonthBusinessDate's extra first-month offset.
    ordinals[months == months[0]] += 1 + np.count_nonzero(interior.dayofweek < 5)
    return {day: np.flatnonzero(ordinals == day) for day in range(1, 30)}


def generate_strategy_configurations(dates=None):
    """Candidate grid; optional dates exclude entry ordinals never observed.

    holding_period is 0..20; stop_loss is a signed sigma multiple (-10..0).
    Without dates, all source entry ordinals 1..29 are candidates. The P&L
    generator subsequently omits candidates with two or fewer observations.
    """
    entry_days = range(1, 30)
    if dates is not None:
        _validate_prices(pd.Series(0., index=dates))
        entry_days = [day for day, positions in _entry_positions(dates).items()
                      if len(positions)]
    configurations = pd.DataFrame(
        itertools.product(entry_days, range(21), range(-10, 1), [-1, 1]),
        columns=["entry_day", "holding_period", "stop_loss", "side"],
    )
    configurations.insert(0, "strategy_id", np.arange(len(configurations)))
    return configurations


def _validate_rule(entry_day, holding_period, stop_loss, side, sigma):
    for name, value, lower, upper in (
        ("entry_day", entry_day, 1, 29),
        ("holding_period", holding_period, 0, 20),
    ):
        if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                or value != int(value) or not lower <= value <= upper):
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    if not np.isfinite(stop_loss) or stop_loss > 0:
        raise ValueError("stop_loss must be a finite, nonpositive sigma multiple.")
    if isinstance(side, (bool, np.bool_)) or side not in (-1, 1):
        raise ValueError("side must be -1 or 1.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")


def _trade_pnl(values, entries, holding_period, stop_threshold, side):
    observations = np.full(len(values), np.nan)
    if not len(entries):
        return observations
    reference = 0
    age = None
    accumulated = 0.
    previous_position = 0
    for i in range(1, len(values)):
        if i >= entries[reference]:
            if i - 1 < entries[reference]:
                age, accumulated = 0, 0.
            if reference < len(entries) - 1:
                while i > entries[reference]:
                    reference += 1
        if age is None:
            continue
        # The source decides today's position before adding today's earned P&L.
        position = side if age < holding_period and accumulated > stop_threshold else 0
        if position != 0 or previous_position != 0:
            observations[i] = previous_position * (values[i] - values[i - 1])
            accumulated += observations[i]
            previous_position = position
        age += 1
    return observations


def run_seasonal_strategy(prices, entry_day, holding_period, stop_loss, side, sigma=1.0):
    """Date-aligned price-unit P&L, with NaN for unrecorded observations.

    Stop thresholds use the known simulation sigma. Entry zeros, exit-day P&L,
    monthly resets and the reference's first-month numbering are preserved.
    """
    values = _validate_prices(prices)
    _validate_rule(entry_day, holding_period, stop_loss, side, sigma)
    observations = _trade_pnl(values, _entry_positions(prices.index)[entry_day],
                              holding_period, stop_loss * sigma, side)
    return pd.Series(observations, index=prices.index, name="pnl")


def generate_strategy_returns(prices, configurations, sigma=1.0):
    """Date-aligned P&L for candidates with more than two recorded observations.

    All input dates are retained for future CSCV alignment; use notna() as the
    observation mask. Unlike the old implementation, inactive days are not zero.
    """
    values = _validate_prices(prices)
    if not isinstance(configurations, pd.DataFrame):
        raise TypeError("configurations must be a Pandas DataFrame.")
    required = {"strategy_id", "entry_day", "holding_period", "stop_loss", "side"}
    if not required.issubset(configurations.columns):
        raise ValueError(f"configurations must contain {sorted(required)}.")
    if configurations.strategy_id.isna().any() or not configurations.strategy_id.is_unique:
        raise ValueError("strategy_id values must be nonmissing and unique.")
    entries = _entry_positions(prices.index)
    recorded = {}
    for strategy in configurations.itertuples(index=False):
        _validate_rule(strategy.entry_day, strategy.holding_period,
                       strategy.stop_loss, strategy.side, sigma)
        observations = _trade_pnl(values, entries[strategy.entry_day],
                                  strategy.holding_period,
                                  strategy.stop_loss * sigma, strategy.side)
        if np.count_nonzero(~np.isnan(observations)) > 2:
            recorded[strategy.strategy_id] = observations
    return pd.DataFrame(recorded, index=prices.index)


# Keep existing callers on the same trade implementation.
generate_strategy_returns_old = generate_strategy_returns


if __name__ == "__main__":
    from synthetic_data import random_walk_seasonal, simulation_dates

    dates = simulation_dates(100)
    prices = pd.Series(random_walk_seasonal(n_days=100), index=dates)
    configurations = generate_strategy_configurations(dates)
    active = configurations.query("holding_period == 4 and stop_loss == -10").head(5)
    pnl = generate_strategy_returns(prices, active)
    assert pnl.index.equals(dates)
    assert pnl.columns.tolist() == active.strategy_id.tolist()
    assert pnl.notna().sum().gt(2).all()
    print("P&L matrix:", pnl.shape, "(NaN denotes an unrecorded observation)")
