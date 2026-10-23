import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis, norm

def sharpe_ratio(returns, risk_free_rate=0.0, annualize=True):
    """
    Calculate the Sharpe Ratio of a trading strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    risk_free_rate : float
        Daily risk-free return.
    annualize : bool
        Whether to annualize the Sharpe Ratio.

    Returns
    -------
    float
        Sharpe Ratio.
    """

    # STEP 1: Remove missing observations
    returns = pd.Series(returns).dropna()

    # STEP 2: Calculate excess returns
    excess_returns = returns - risk_free_rate

    # STEP 3: Calculate average excess return
    mean_return = excess_returns.mean()

    # STEP 4: Calculate return volatility
    volatility = excess_returns.std(ddof=1)

    # STEP 5: Handle insufficient data or zero volatility
    if len(returns) < 2 or volatility == 0:
        return np.nan

    # STEP 6: Calculate the daily Sharpe Ratio
    sr = mean_return / volatility

    # STEP 7: Annualize if requested
    if annualize:
        sr *= np.sqrt(252)

    return sr

################################################################################################################################################################


def sortino_ratio(
    returns,
    risk_free_rate=0.0,
    minimum_acceptable_return=0.0,
    annualize=True
):
    """
    Calculate the Sortino Ratio using downside deviation.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    risk_free_rate : float
        Daily risk-free return.
    minimum_acceptable_return : float
        Minimum acceptable daily return (MAR).
    annualize : bool
        Whether to annualize the Sortino Ratio.

    Returns
    -------
    float
        Sortino Ratio.
    """

    # STEP 1: Remove missing values
    returns = pd.Series(returns).dropna()

    # STEP 2: Check that observations exist
    if len(returns) < 2:
        return np.nan

    # STEP 3: Calculate average excess return
    mean_excess_return = returns.mean() - risk_free_rate

    # STEP 4: Calculate returns below the target
    downside_returns = np.minimum(
        returns - minimum_acceptable_return,
        0
    )

    # STEP 5: Calculate downside deviation
    downside_deviation = np.sqrt(
        np.mean(downside_returns ** 2)
    )

    # STEP 6: Avoid division by zero
    if downside_deviation == 0:
        return np.nan

    # STEP 7: Calculate the Sortino Ratio
    sortino = mean_excess_return / downside_deviation

    # STEP 8: Annualize if requested
    if annualize:
        sortino *= np.sqrt(252)

    return sortino

################################################################


def maximum_drawdown(returns):
    """
    Calculate the Maximum Drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.

    Returns
    -------
    float
        Maximum peak-to-trough percentage decline.
    """

    # STEP 1: Remove missing values
    returns = pd.Series(returns).dropna()

    # Handle empty data
    if returns.empty:
        return np.nan

    # STEP 2: Calculate cumulative portfolio value
    portfolio_value = (1 + returns).cumprod()

    # Include the initial portfolio value of 1
    portfolio_value = pd.concat([
        pd.Series([1.0]),
        portfolio_value.reset_index(drop=True)
    ], ignore_index=True)

    # STEP 3: Find the highest portfolio value reached so far
    running_peak = portfolio_value.cummax()

    # STEP 4: Calculate drawdown at each observation
    drawdown = 1 - portfolio_value / running_peak

    # STEP 5: Find the largest drawdown
    max_drawdown = drawdown.max()

    return max_drawdown


################################################################################################################################################################


def probabilistic_sharpe_ratio(returns, benchmark_sr=0.0):
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    benchmark_sr : float
        Benchmark Sharpe Ratio, in daily units.

    Returns
    -------
    float
        Probability between 0 and 1.
    """

    # STEP 1: Remove missing observations
    returns = pd.Series(returns).dropna()

    # Number of observations
    n = len(returns)

    # Require sufficient observations for higher moments
    if n < 4:
        return np.nan

    # STEP 2: Calculate the daily Sharpe Ratio
    sr = sharpe_ratio(
        returns,
        annualize=False
    )

    if not np.isfinite(sr):
        return np.nan

    # STEP 3: Calculate skewness of returns
    gamma3 = skew(returns, bias=False)

    # STEP 4: Calculate kurtosis (Pearson convention)
    gamma4 = kurtosis(
        returns,
        fisher=False,
        bias=False
    )

    # STEP 5: Calculate the denominator
    variance_term = (
        1
        - gamma3 * sr
        + ((gamma4 - 1) / 4) * sr**2
    )

    # Check that the denominator is valid
    if not np.isfinite(variance_term) or variance_term <= 0:
        return np.nan

    # STEP 6: Calculate the PSR test statistic
    z_score = (
        (sr - benchmark_sr) * np.sqrt(n - 1)
        / np.sqrt(variance_term)
    )

    # STEP 7: Convert the statistic into a probability
    psr = norm.cdf(z_score)

    return float(psr)


################################################################################################################################################################


def evaluate_strategies(strategy_matrix, benchmark_sr=0.0):
    """
    Calculate performance metrics for every strategy.

    Parameters
    ----------
    strategy_matrix : pd.DataFrame
        Daily returns with one strategy per column.
    benchmark_sr : float
        Benchmark daily Sharpe Ratio for PSR.

    Returns
    -------
    pd.DataFrame
        Performance metrics for each strategy.
    """

    # Store the results for each strategy
    results = []

    # Loop through each strategy column
    for strategy_id in strategy_matrix.columns:

        # Extract this strategy's daily returns
        returns = strategy_matrix[strategy_id]

        # Calculate the four performance metrics
        results.append({
            "strategy_id": strategy_id,
            "sharpe": sharpe_ratio(returns),
            "sortino": sortino_ratio(returns),
            "max_drawdown": maximum_drawdown(returns),
            "psr": probabilistic_sharpe_ratio(
                returns,
                benchmark_sr=benchmark_sr
            )
        })

    # Convert the results into a table
    return pd.DataFrame(results)


#################################################################################

if __name__ == "__main__":

    # Small example of five daily strategy returns
    test_returns = pd.Series([
        0.01,
        -0.005,
        0.015,
        -0.002,
        0.008
    ])

    # Calculate the daily Sharpe Ratio
    daily_sr = sharpe_ratio(
        test_returns,
        annualize=False
    )

    # Calculate the annualized Sharpe Ratio
    annual_sr = sharpe_ratio(
        test_returns,
        annualize=True
    )

    print("Daily Sharpe Ratio:", round(daily_sr, 4))
    print("Annualized Sharpe Ratio:", round(annual_sr, 4))


    # Test the Sortino Ratio using the same returns
    daily_sortino = sortino_ratio(
        test_returns,
        annualize=False
    )

    annual_sortino = sortino_ratio(
        test_returns,
        annualize=True
    )

    print("\nSortino Ratio Results:")
    print("Daily Sortino Ratio:", round(daily_sortino, 4))
    print("Annualized Sortino Ratio:", round(annual_sortino, 4))


    # Test Maximum Drawdown using the same five returns
    mdd = maximum_drawdown(test_returns)

    print("\nMaximum Drawdown Results:")
    print("Maximum Drawdown:", round(mdd, 4))
    print("Maximum Drawdown (%):", round(mdd * 100, 2))


    # Test the Probabilistic Sharpe Ratio
    psr = probabilistic_sharpe_ratio(
        test_returns,
        benchmark_sr=0.0
    )

    print("\nProbabilistic Sharpe Ratio Results:")
    print("PSR:", round(psr, 4))
    print("PSR (%):", round(psr * 100, 2))




