from scipy.stats import linregress
import numpy as np


def _finite_columns(frame, columns):
    values = frame[columns].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"Expected nonempty, finite {columns}.")
    return values


def compute_pbo(cscv_results):
    logits = _finite_columns(cscv_results, ["logit"])
    return (logits < 0).mean()


def performance_degradation(cscv_results):
    is_performance, oos_performance = _finite_columns(
        cscv_results, ["performance_is", "performance_oos"]
    ).T
    if len(is_performance) < 2 or np.ptp(is_performance) == 0:
        return {"slope": np.nan, "intercept": np.nan, "R^2": np.nan, "defined": False}

    regression = linregress(is_performance, oos_performance)

    result = {
        "slope": regression.slope,
        "intercept": regression.intercept,
        "R^2": regression.rvalue ** 2 if np.ptp(oos_performance) > 0 else np.nan,
        "defined": True,
    }

    return result


def probability_of_loss(cscv_results):
    oos_scores = _finite_columns(cscv_results, ["performance_oos"])
    return (oos_scores < 0).mean()

def stochastic_dominance(cscv_results, benchmark_scores=None):
    """Empirical dominance against split means, or an explicit score population.

    Each empirical distribution has its own denominator, so a pooled benchmark
    need not have the same number of observations as the selected distribution.
    """
    selected = _finite_columns(cscv_results, ["performance_oos"])[:, 0]
    if benchmark_scores is None:
        benchmark = _finite_columns(cscv_results, ["benchmark_oos"])[:, 0]
    else:
        benchmark = np.asarray(benchmark_scores, dtype=float)
        if benchmark.ndim != 1 or not len(benchmark) or not np.isfinite(benchmark).all():
            raise ValueError("Expected a nonempty, finite 1D benchmark score population.")

    # Evaluate empirical CDFs at all observed values
    x = np.unique(np.concatenate([selected, benchmark]))

    selected_sorted = np.sort(selected)
    benchmark_sorted = np.sort(benchmark)

    F_selected = np.searchsorted(
        selected_sorted, x, side="right"
    ) / len(selected)

    F_benchmark = np.searchsorted(
        benchmark_sorted, x, side="right"
    ) / len(benchmark)

    # First-order stochastic dominance
    difference = F_benchmark - F_selected

    tol = 1e-10

    fsd = (
        np.all(difference >= -tol)
        and np.any(difference > tol)
    )

    # Second-order stochastic dominance
    sd2 = np.zeros(len(x))

    sd2[1:] = np.cumsum(difference[:-1] * np.diff(x))

    ssd = (
        np.all(sd2 >= -tol)
        and np.any(sd2 > tol)
    )

    return {
        "first_order_dominance": bool(fsd),
        "second_order_dominance": bool(ssd),
        "x": x,
        "cdf_selected": F_selected,
        "cdf_benchmark": F_benchmark,
        "sd2": sd2
    }
