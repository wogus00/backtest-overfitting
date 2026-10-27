import pandas as pd
import numpy as np
from pathlib import Path
from synthetic_data import random_walk, random_walk_seasonal
from strategies import generate_strategy_configurations, generate_strategy_returns
from performance_metrics import sharpe_ratio, sortino_ratio, evaluate_strategies, maximum_drawdown, probabilistic_sharpe_ratio
from cscv import cscv
from overfit_statistics import performance_degradation, stochastic_dominance, compute_pbo, probability_of_loss
from cscv_validation import (validate_effect_of_S, 
                             validate_effect_of_N, 
                             validate_temporal_dependence, 
                             validate_autocorrelation_preservation, 
                             validate_pbo_accuracy, 
                             validate_model_free)




def get_prices(prices_dir, seed=42):
    prices_no_signal = random_walk(n_days=1000, sigma=0.01, seed=seed)
    prices_seasonal = random_walk_seasonal(n_days=1000, sigma=0.01, seed=seed)
    prices_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"price": prices_no_signal}).to_parquet(prices_dir / "no_signal.parquet", index=False)
    pd.DataFrame({"price": prices_seasonal}).to_parquet(prices_dir / "seasonal.parquet", index=False)

    return prices_no_signal, prices_seasonal


def get_strategies(directory, price_dir):
    configurations_path = directory / "strategy_configurations.parquet"

    returns_dir = directory / "returns"

    configurations = generate_strategy_configurations()

    prices_no_signal = pd.read_parquet(price_dir / "no_signal.parquet")["price"].to_numpy()
    prices_seasonal = pd.read_parquet(price_dir / "seasonal.parquet")["price"].to_numpy()


    dates = pd.bdate_range(start="2020-01-01", periods=len(prices_no_signal))
    prices_no_signal = pd.Series(prices_no_signal, index=dates, name="price")
    prices_seasonal = pd.Series(prices_seasonal, index=dates, name="price")

    returns_no_signal = generate_strategy_returns(prices_no_signal, configurations)
    returns_seasonal = generate_strategy_returns(prices_seasonal, configurations)

    returns_dir.mkdir(parents=True, exist_ok=True)

    configurations.to_parquet(configurations_path, index=False)
    returns_no_signal.to_parquet(returns_dir / "no_signal.parquet")
    returns_seasonal.to_parquet(returns_dir / "seasonal.parquet")

    return configurations, returns_no_signal, returns_seasonal


def measure_performance(returns, configurations):
    performance = evaluate_strategies(returns)

    return performance.merge(configurations,on="strategy_id",how="left",validate="one_to_one")



def score_columns(matrix, metric=sharpe_ratio):
    matrix = np.asarray(matrix, dtype=float)

    scores = np.array([metric(matrix[:, column]) for column in range(matrix.shape[1])])

    return scores

def run_cscv(returns, S=8, metric=sharpe_ratio):
    results = cscv(returns.to_numpy(),
                   S=S,
                   metric=lambda matrix: score_columns(matrix, metric=metric))

    results["strategy_id"] = returns.columns.to_numpy()[results["selected_idx"].to_numpy()]

    return results

def compute_statistics(cscv_results, metric_name, benchmark=None):
    degradation = performance_degradation(cscv_results)
    dominance = stochastic_dominance(cscv_results)

    summary = {"metric": metric_name,
               "pbo": compute_pbo(cscv_results),
               "degradation_slope": degradation["slope"],
               "degradation_intercept": degradation["intercept"],
               "degradation_r2": degradation["R^2"],
               "probability_of_compounded_loss": probability_of_loss(cscv_results),
               "first_order_dominance_vs_split_mean": dominance["first_order_dominance"],
               "second_order_dominance_vs_split_mean": dominance["second_order_dominance"]}

    if benchmark is not None:
        summary["oos_score_benchmark"] = benchmark
        summary["probability_below_oos_benchmark"] = (cscv_results["performance_oos"] < benchmark).mean()

    return summary, dominance

def run_validation(
    returns,
    S_values,
    N_values,
    S=8,
    estimated_pbo=None,
    reference_pbo=None,
):
    if (estimated_pbo is None) != (reference_pbo is None):
        raise ValueError(
            "Provide both estimated_pbo and reference_pbo, or neither."
        )

    validation = {
        "effect_of_S": validate_effect_of_S(
            returns, S_values=S_values
        ),
        "effect_of_N": validate_effect_of_N(
            returns, N_values=N_values
        ),
        "temporal_dependence": validate_temporal_dependence(
            returns, S=S
        ),
        "autocorrelation": validate_autocorrelation_preservation(
            T=len(returns), S=S
        ),
        "model_free": validate_model_free(
            T=len(returns), S=S
        ),
    }

    if estimated_pbo is not None:
        validation["pbo_accuracy"] = validate_pbo_accuracy(
            estimated_pbo, reference_pbo
        )

    return validation


def save_results(
                output_directory,
                dataset_name,
                 performance,
                 cscv_results,
                 summary,
                 dominance,
                 validation):
    output_dir = Path(output_directory) / dataset_name / summary["metric"]
    output_dir.mkdir(parents=True, exist_ok=True)

    performance.to_parquet(output_dir / "performance.parquet", index=False)
    cscv_results.to_parquet(output_dir / "cscv.parquet", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    pd.DataFrame({"score": dominance["x"], 
                  "cdf_selected": dominance["cdf_selected"],
                  "cdf_split_mean": dominance["cdf_benchmark"],
                  "integrated_cdf_difference": dominance["sd2"]}).to_parquet(output_dir / "dominance.parquet", index=False)

    for name, result in validation.items():
        # PBO accuracy returns a dict; other checks return DataFrames.
        table = pd.DataFrame([result]) if isinstance(result, dict) else result
        table.to_parquet(output_dir / f"validation_{name}.parquet",index=False)

    return output_dir


def main(experiment_name, n_times=5, real_data=False, base_seed=42, S=8):
    metrics = {
        "sharpe": sharpe_ratio,
        # "sortino": sortino_ratio,
        # "max_drawdown": maximum_drawdown,
        # "probabilistic_sharpe": probabilistic_sharpe_ratio
    }

    print(f"Experiment: {experiment_name}")
    DATA_DIR = Path(f"data/{experiment_name}")
    RESULT_DIR = Path(f"result/{experiment_name}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    if not real_data:
        for i in range(n_times):

            seed = base_seed + i
            curr_data_dir = DATA_DIR / f"seed_{seed}"
            curr_results_dir = RESULT_DIR / f"seed_{seed}"

            print(f"Running simulation {i+1}/{n_times}, seed={seed}")

            # 1. Generate synthetic prices using the current seed
            get_prices(curr_data_dir, seed=seed)

            # 2. Generate strategy returns for both datasets
            configurations, returns_no_signal, returns_seasonal = get_strategies(curr_results_dir, curr_data_dir)
            datasets = {"no_signal": returns_no_signal,"seasonal": returns_seasonal}


            # for each dataset, compute performance metrics, run CSCV, and save results
            for dataset_name, returns in datasets.items():
                # compute performance metrics
                performance = measure_performance(returns, configurations)   
                validation = run_validation(
                    returns,
                    S_values=[
                        s for s in [4, 8, 16]
                        if s <= len(returns) and len(returns) % s == 0],
                    N_values=sorted({
                        n for n in [10, 50, 100, 500, 1000, returns.shape[1]]
                        if 2 <= n <= returns.shape[1]}),
                    S=S)

                # run for each performance metric
                for performance_metric, metric_func in metrics.items():

                    # run CSCV and compute statistics for each performance metric
                    cscv_results = run_cscv(
                        returns,
                        S=S,
                        metric=metric_func)

                    summary, dominance = compute_statistics(cscv_results, metric_name=performance_metric, benchmark=0.0)
                    output_directory = curr_results_dir
                    output_dir = save_results(
                        output_directory,
                        dataset_name=dataset_name,
                        performance=performance,
                        cscv_results=cscv_results,
                        summary=summary,
                        dominance=dominance,
                        validation=validation,
                    )

                    results[(seed, dataset_name, performance_metric)] = summary
                    print(f"Saved to {output_dir}")
                
    
    else:
        # handle real data case
        pass # for now


    print("\nAll simulations completed.")
    print("Final results:")

    for (seed, dataset_name, metric_name), result in results.items():
        print(f"\nSeed: {seed}, Dataset: {dataset_name}, Metric: {metric_name}")
        print(pd.Series(result).to_string())


    


if __name__ == "__main__":

    main(
        experiment_name="test_experiment",
        n_times=2,
        real_data=False,
        base_seed=42,
        S=8
    )
