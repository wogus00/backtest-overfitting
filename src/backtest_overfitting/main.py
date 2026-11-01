import pandas as pd
import numpy as np
from pathlib import Path
from . import paths
from .synthetic_data import random_walk, random_walk_seasonal, simulation_dates
from .strategies import generate_strategy_configurations, generate_strategy_returns
from .performance_metrics import sharpe_ratio, column_sharpe_ratios, evaluate_pnl_strategies, calendar_day_weights
from .cscv import cscv, split_blocks
from .overfit_statistics import performance_degradation, stochastic_dominance, compute_pbo, probability_of_loss
from .cscv_validation import (validate_effect_of_S, 
                             validate_effect_of_N, 
                             validate_temporal_dependence, 
                             validate_pbo_accuracy)



def get_prices(prices_dir, seed=42, n_days=1000):
    dates = simulation_dates(n_days)
    prices_no_signal = random_walk(n_days=n_days, sigma=1.0, seed=seed)
    prices_seasonal = random_walk_seasonal(n_days=n_days, sigma=1.0, seed=seed)
    prices_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"date": dates, "price": prices_no_signal}).to_parquet(prices_dir / "no_signal.parquet", index=False)
    pd.DataFrame({"date": dates, "price": prices_seasonal}).to_parquet(prices_dir / "seasonal.parquet", index=False)

    return prices_no_signal, prices_seasonal


def get_strategies(directory, price_dir):
    configurations_path = directory / "strategy_configurations.parquet"

    returns_dir = directory / "returns"

    ordinary = pd.read_parquet(price_dir / "no_signal.parquet")
    seasonal = pd.read_parquet(price_dir / "seasonal.parquet")
    dates = pd.DatetimeIndex(ordinary["date"])
    if not dates.equals(pd.DatetimeIndex(seasonal["date"])):
        raise ValueError("Scenario dates must align.")
    configurations = generate_strategy_configurations(dates)
    prices_no_signal = pd.Series(ordinary["price"].to_numpy(), index=dates, name="price")
    prices_seasonal = pd.Series(seasonal["price"].to_numpy(), index=dates, name="price")

    returns_no_signal = generate_strategy_returns(prices_no_signal, configurations)
    returns_seasonal = generate_strategy_returns(prices_seasonal, configurations)

    returns_dir.mkdir(parents=True, exist_ok=True)

    configurations.to_parquet(configurations_path, index=False)
    returns_no_signal.to_parquet(returns_dir / "no_signal.parquet")
    returns_seasonal.to_parquet(returns_dir / "seasonal.parquet")

    return configurations, returns_no_signal, returns_seasonal


def measure_performance(returns, configurations):
    performance = evaluate_pnl_strategies(returns)

    return performance.merge(configurations,on="strategy_id",how="left",validate="one_to_one")



def score_columns(matrix, metric=sharpe_ratio):
    matrix = np.asarray(matrix, dtype=float)
    if metric is sharpe_ratio:
        return column_sharpe_ratios(matrix)

    scores = np.array([metric(matrix[:, column]) for column in range(matrix.shape[1])])

    return scores

def run_cscv(returns, S=8, metric=None):
    if metric is None:
        results = cscv(returns.to_numpy(), S=S,
                       calendar_days=calendar_day_weights(returns.index))
        eligibility = results.attrs["eligibility"]
        eligibility["strategy_id"] = returns.columns.to_numpy()[eligibility.column_index]
    else:
        # Explicit custom metrics retain the original finite-return interface.
        results = cscv(returns.to_numpy(), S=S,
                       metric=lambda matrix: score_columns(matrix, metric=metric))

    results["strategy_id"] = returns.columns.to_numpy()[results["selected_idx"].to_numpy()]

    return results

def compute_statistics(cscv_results, metric_name, benchmark=None):
    pbo = compute_pbo(cscv_results)
    degradation = performance_degradation(cscv_results)
    dominance = stochastic_dominance(cscv_results)

    summary = {"metric": metric_name,
               "pbo": pbo,
               "degradation_slope": degradation["slope"],
               "degradation_intercept": degradation["intercept"],
               "degradation_r2": degradation["R^2"],
               "degradation_defined": degradation["defined"],
               "probability_of_loss": probability_of_loss(cscv_results),
               "first_order_dominance_vs_split_mean": dominance["first_order_dominance"],
               "second_order_dominance_vs_split_mean": dominance["second_order_dominance"]}

    if benchmark is not None:
        if not np.isfinite(benchmark):
            raise ValueError("OOS score benchmark must be finite.")
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
    """Section 4 structural diagnostics; optional errors need external references."""
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
    if (output_dir / "performance.csv").exists():
        performance.to_csv(output_dir / "performance.csv", index=False)
    if "eligibility" in cscv_results.attrs:
        cscv_results.attrs["eligibility"].to_parquet(output_dir / "eligibility.parquet", index=False)
    records = cscv_results.copy()
    records.attrs = {}
    records.to_parquet(output_dir / "cscv.parquet", index=False)
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


def main(experiment_name, n_times=1, real_data=False, base_seed=1, S=8, n_days=1000):
    import json

    if real_data:
        raise NotImplementedError("This runner implements the synthetic P&L experiment.")
    simulation_dates(n_days)
    split_blocks(np.empty((n_days, 1)), S)
    if isinstance(n_times, bool) or not isinstance(n_times, int) or n_times < 1:
        raise ValueError("n_times must be a positive integer.")
    results = {}
    for seed in range(base_seed, base_seed + n_times):
        current_data = paths.data_dir(experiment_name, seed)
        current_processed = paths.processed_dir(experiment_name, seed)
        current_results = paths.result_dir(experiment_name, seed)
        print(f"Experiment: {experiment_name}; seed={seed}; T={n_days}; S={S}")
        get_prices(current_data, seed=seed, n_days=n_days)
        configurations, ordinary, seasonal = get_strategies(current_processed, current_data)
        metadata = dict(experiment=experiment_name, seed=seed, S=S, T=n_days, sigma=1., units="price_unit_pnl",
                        source="https://www.quantresearch.org/CSCV_3.py.txt",
                        calendar="Monday-Thursday from 2000-01-01 (literal source)",
                        rng="random.Random(seed).gauss; paired shocks across scenarios",
                        seasonality="source numBusinessDays <= 5; shift 0.25 sigma",
                        annualization="365.25; each row owns days until the next row; last row owns one day",
                        eligibility="fixed universe with >2 records and positive variance in every half-split",
                        dominance_benchmark="mean OOS Sharpe across eligible strategies per split",
                        reproduction="source-style simulation; original figure seed and CSCV conventions unverified")
        current_results.mkdir(parents=True, exist_ok=True)
        (current_results / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for name, pnl in (("no_signal", ordinary), ("seasonal", seasonal)):
            cscv_results = run_cscv(pnl, S=S)
            eligibility = cscv_results.attrs["eligibility"]
            ids = eligibility.loc[eligibility.eligible, "strategy_id"].tolist()
            eligible_pnl = pnl[ids]
            performance = measure_performance(eligible_pnl, configurations)
            validation = run_validation(
                eligible_pnl,
                S_values=[s for s in sorted({4, 8, 16, S}) if len(pnl) % s == 0],
                N_values=sorted({n for n in [10, 50, 100, 500, 1000, len(ids)] if n <= len(ids)}),
                S=S)
            summary, dominance = compute_statistics(cscv_results, "sharpe", benchmark=0.)
            summary.update(S=S, T=len(pnl), n_strategies=len(ids),
                           n_excluded_strategies=len(pnl.columns) - len(ids), units="price_unit_pnl")
            output = save_results(paths.tables_dir(experiment_name, seed), name, performance, cscv_results,
                                  summary, dominance, validation)
            results[(seed, name)] = summary
            print(f"Saved {output}: PBO={summary['pbo']:.2%}, loss={summary['probability_of_loss']:.2%}")
    return results


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Run the source-style seasonal P&L experiment.")
    parser.add_argument("--experiment", default="paper_source_S8")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-times", type=int, default=1)
    parser.add_argument("--S", type=int, default=8)
    parser.add_argument("--n-days", type=int, default=1000)
    args = parser.parse_args()
    main(args.experiment, n_times=args.n_times, base_seed=args.seed, S=args.S, n_days=args.n_days)


if __name__ == "__main__":
    cli()
