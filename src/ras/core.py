from numbers import Real

import numpy as np



def rademacher_complexity(X, B=1000, seed=42):
    """Estimate E_epsilon[max_n(epsilon @ X[:, n] / T)] using B draws.
    """
    rng = np.random.default_rng(seed)
    epsilon = rng.choice([-1.0, 1.0], size=(B, X.shape[0]))
    with np.errstate(over="raise", invalid="raise"):
        maxima = np.max(epsilon @ (X / X.shape[0]), axis=1)
        return float(np.mean(maxima))


def ras_sharpe(X, delta=0.1, B=1000, seed=42):
    """Return the components of the nonannualized RAS Sharpe lower bound.
    """
    T, N = X.shape
    complexity = rademacher_complexity(X, B=B, seed=seed)
    log_confidence = np.log(2.0) - np.log(delta)
    estimation_error = float(
        3 * np.sqrt(2 * log_confidence / T)
        + np.sqrt(2 * (log_confidence + np.log(N)) / T)
    )
    data_snooping = 2 * complexity
    haircut = data_snooping + estimation_error
    with np.errstate(over="raise", invalid="raise"):
        empirical_sharpe = np.sum(X / T, axis=0)
        lower_bound = empirical_sharpe - haircut
    return {
        "empirical_sharpe": empirical_sharpe,
        "rademacher_complexity": complexity,
        "data_snooping": data_snooping,
        "estimation_error": estimation_error,
        "haircut": haircut,
        "lower_bound": lower_bound,
    }
