
import itertools
import numpy as np
import pandas as pd


# 1. GENERATE STRATEGY CONFIGURATIONS


def generate_strategy_configurations():
    """
    Generate 8,800 seasonal trading strategy configurations.

    Parameters:
        entry_day      - Trading day of the month to enter
        holding_period - Maximum number of trading days to hold
        stop_loss      - Volatility-based stop-loss multiplier
        side           - +1 for long, -1 for short

    Returns:
        DataFrame containing all strategy configurations.
    """

    # Define the parameter ranges
    entry_days = range(1, 23)
    holding_periods = range(1, 21)
    stop_losses = range(0, 10)
    sides = [-1, 1]

    # Generate every possible parameter combination
    combinations = itertools.product(
        entry_days,
        holding_periods,
        stop_losses,
        sides
    )

    # Store each configuration as one row
    configurations = pd.DataFrame(
        combinations,
        columns=[
            "entry_day",
            "holding_period",
            "stop_loss",
            "side"
        ]
    )

    # Assign a unique strategy ID for merging results
    configurations.insert(
        0,
        "strategy_id",
        np.arange(len(configurations))
    )

    return configurations

# 2. EXECUTE ONE SEASONAL TRADING STRATEGY


def run_seasonal_strategy(
    prices,
    entry_day,
    holding_period,
    stop_loss,
    side
):
    """
    Simulate one monthly seasonal trading strategy.

    Enter a long or short position on the chosen trading day
    of each month and hold for at most holding_period returns,
    exiting earlier if the stop-loss threshold is reached.

    Assumptions:
        - Trades enter at the closing price of entry_day.
        - Stop-loss distance uses the previous 20 daily returns.
        - Stop-loss multiplier is fixed when entering the trade.
        - Trades are executed at observed closing prices.
        - No transaction costs or slippage.
        - Only one position can be open at a time.

    Returns:
        A Pandas Series of daily strategy returns.
    """

    # Ensure prices are a numeric Pandas Series
    prices = pd.Series(prices, copy=True).astype(float)

    # Calculate the underlying asset's daily returns
    asset_returns = prices.pct_change().fillna(0.0)

    # Use only past returns to calculate volatility
    volatility = asset_returns.rolling(20).std().shift(1)

    # Start with zero returns when not invested
    strategy_returns = pd.Series(
        0.0,
        index=prices.index,
        dtype=float
    )

    # Trading-day number within each calendar month
    trading_days = (
        pd.Series(1, index=prices.index)
        .groupby(prices.index.to_period("M"))
        .cumsum()
    )

    # Prevent trades from overlapping
    last_exit = -1

    # Examine each month separately
    for month in prices.index.to_period("M").unique():

        # Find the row positions belonging to this month
        month_indices = np.flatnonzero(
            prices.index.to_period("M") == month
        )

        # Skip months that don't have the requested entry day
        if entry_day > len(month_indices):
            continue

        # Identify the entry position
        entry_idx = month_indices[entry_day - 1]

        # Skip if an earlier trade is still active
        if entry_idx <= last_exit:
            continue

        # Need historical observations to estimate volatility
        if pd.isna(volatility.iloc[entry_idx]):
            continue

        entry_price = prices.iloc[entry_idx]

        # Fix the stop-loss threshold at the entry date
        stop_threshold = (
            float(stop_loss) * volatility.iloc[entry_idx]
        )

        # Limit the trade to the requested holding period
        exit_limit = min(
            entry_idx + int(holding_period),
            len(prices) - 1
        )

        # Examine subsequent daily closes
        for i in range(entry_idx + 1, exit_limit + 1):

            # Daily return earned in the position
            daily_strategy_return = (
                int(side) * asset_returns.iloc[i]
            )

            strategy_returns.iloc[i] = daily_strategy_return

            # Approximate cumulative directional price return
            cumulative_trade_return = (
                int(side) *
                (prices.iloc[i] / entry_price - 1)
            )

            # Stop trading when the loss threshold is reached
            if cumulative_trade_return <= -stop_threshold:
                last_exit = i
                break

        else:
            # The position reached its maximum holding period
            last_exit = exit_limit

    return strategy_returns


# 3. GENERATE RETURNS FOR ALL STRATEGIES


def generate_strategy_returns_old(prices, configurations):
    """
    Run all seasonal configurations on one price series.

    Parameters:
        prices:
            Pandas Series indexed by business dates.

        configurations:
            DataFrame returned by generate_strategy_configurations().

    Returns:
        T x N DataFrame of daily strategy returns,
        with strategy_id values as column labels.
    """

    # Verify the input price series has a DatetimeIndex
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a Pandas Series.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")

    # Store the daily return Series for every strategy
    all_returns = {}

    # Iterate over the configuration table
    for _, strategy in configurations.iterrows():

        # Extract this strategy's unique identifier
        strategy_id = int(strategy["strategy_id"])

        # Run the strategy with its own parameters
        returns = run_seasonal_strategy(
            prices=prices,
            entry_day=int(strategy["entry_day"]),
            holding_period=int(strategy["holding_period"]),
            stop_loss=float(strategy["stop_loss"]),
            side=int(strategy["side"])
        )

        # Save the return Series using strategy_id
        all_returns[strategy_id] = returns

    # Combine the individual Series into a return matrix
    strategy_matrix = pd.DataFrame(
        all_returns,
        index=prices.index
    )

    return strategy_matrix

def generate_strategy_returns(prices, configurations):
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a Pandas Series.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")

    # Precompute data once
    prices_array = prices.to_numpy(dtype=float)

    asset_returns = prices.astype(float).pct_change().fillna(0.0)
    returns_array = asset_returns.to_numpy()

    volatility = (
        asset_returns.rolling(20).std().shift(1).to_numpy()
    )

    # Precompute monthly indices once
    months = prices.index.to_period("M")
    month_indices = [
        np.flatnonzero(months == month)
        for month in months.unique()
    ]

    T = len(prices)
    N = len(configurations)

    # Preallocate return matrix
    strategy_matrix = np.zeros((T, N), dtype=float)

    for col, strategy in enumerate(configurations.itertuples(index=False)):
        entry_day = int(strategy.entry_day)
        holding_period = int(strategy.holding_period)
        stop_loss = float(strategy.stop_loss)
        side = int(strategy.side)

        last_exit = -1

        for indices in month_indices:

            if entry_day > len(indices):
                continue

            entry_idx = indices[entry_day - 1]

            if entry_idx <= last_exit:
                continue

            if np.isnan(volatility[entry_idx]):
                continue

            entry_price = prices_array[entry_idx]
            stop_threshold = stop_loss * volatility[entry_idx]

            exit_limit = min(
                entry_idx + holding_period,
                T - 1
            )

            for i in range(entry_idx + 1, exit_limit + 1):

                strategy_matrix[i, col] = side * returns_array[i]

                cumulative_return = side * (
                    prices_array[i] / entry_price - 1
                )

                if cumulative_return <= -stop_threshold:
                    last_exit = i
                    break

            else:
                last_exit = exit_limit

    return pd.DataFrame(
        strategy_matrix,
        index=prices.index,
        columns=configurations["strategy_id"].to_numpy()
    )


# 4. SMALL TEST (RUNS ONLY WHEN EXECUTING THIS FILE)

if __name__ == "__main__":

    # Generate the full configuration grid
    configurations = generate_strategy_configurations()

    print("Total configurations:", len(configurations))
    print(configurations.head())

    # Generate a small synthetic price Series
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        start="2020-01-01",
        periods=100
    )

    daily_returns = rng.normal(0, 0.01, len(dates))

    prices = pd.Series(
        100 * np.cumprod(1 + daily_returns),
        index=dates,
        name="price"
    )

    # Test only five configurations to keep this quick
    test_configs = configurations.head(5)

    returns = generate_strategy_returns(
        prices,
        test_configs
    )

    print("\nTest return matrix shape:", returns.shape)
    print("Strategy IDs:", returns.columns.tolist())
    print("Missing values:", returns.isna().sum().sum())

    assert len(configurations) == 8800
    assert returns.shape == (100, 5)
    assert returns.columns.tolist() == [0, 1, 2, 3, 4]

    print("\nAll strategy tests passed!")
