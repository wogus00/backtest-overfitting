
import numpy as np
import pandas as pd

from synthetic_data import random_walk, random_walk_seasonal
from strategies import generate_strategy_returns
from cscv import cscv

from performance_metrics import sharpe_ratio, sortino_ratio, negative_max_drawdown, probabilistic_sharpe_ratio

from overfit_statistics import compute_pbo, performance_degradation, probability_of_loss, stochastic_dominance


# Experiment configuration
N_DAYS = 1008
S = 16
SIGMA = 0.01
SEED = 42

DATASETS = {"random_walk": random_walk, 
            "random_walk_seasonal": random_walk_seasonal}

METRICS = {"sharpe": sharpe_ratio,
           "sortino": sortino_ratio,
           "max_drawdown": negative_max_drawdown,
           "psr": probabilistic_sharpe_ratio}


def run_experiment(dataset_name, data_generator, metric_name, metric):
    prices = data_generator(n_days=N_DAYS,sigma=SIGMA,seed=SEED)

    # 2. Generate T x N strategy-return matrix
    M = generate_strategy_returns(prices)

    # 3. Run CSCV
    cscv_results = cscv(M, S=S, metric=metric)

    # 4. Calculate overfitting statistics
    degradation = performance_degradation(cscv_results)
    dominance = stochastic_dominance(cscv_results)

    return {
        "dataset": dataset_name,
        "metric": metric_name,
        "n_strategies": M.shape[1],
        "pbo": compute_pbo(cscv_results),
        "degradation_slope": degradation["slope"],
        "degradation_r2": degradation["R^2"],
        "probability_of_loss": probability_of_loss(cscv_results),
        "first_order_dominance": dominance["first_order_dominance"],
        "second_order_dominance": dominance["second_order_dominance"]}


def main():
    results = []

    for dataset_name, generator in DATASETS.items():
        for metric_name, metric in METRICS.items():
            print(f"Running {dataset_name} with {metric_name}...")

            result = run_experiment(
                dataset_name,
                generator,
                metric_name,
                metric,
            )

            results.append(result)

    summary = pd.DataFrame(results)

    print("\nExperiment Results")
    print(summary.to_string(index=False))

    summary.to_csv("experiment_results.csv", index=False)


if __name__ == "__main__":
    main()
