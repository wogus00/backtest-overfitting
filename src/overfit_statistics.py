from scipy.stats import linregress
import numpy as np


def compute_pbo(cscv_results):
    logits = cscv_results["logit"]
    return (logits < 0).mean()


def performance_degradation(cscv_results):
    is_performance = cscv_results["performance_is"]
    oos_performance = cscv_results["performance_oos"]

    regression = linregress(is_performance, oos_performance)

    result = {
        "slope": regression.slope,
        "intercept": regression.intercept,
        "R^2": regression.rvalue ** 2
    }

    return result


def probability_of_loss(cscv_results):
    oos_returns = cscv_results["oos_total_return"]
    return (oos_returns < 0).mean()

def stochastic_dominance(cscv_results):
    selected = cscv_results["performance_oos"].to_numpy()
    benchmark = cscv_results["benchmark_oos"].to_numpy()

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

    for i in range(1, len(x)):
        width = x[i] - x[i - 1]
        sd2[i] = sd2[i - 1] + difference[i - 1] * width

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