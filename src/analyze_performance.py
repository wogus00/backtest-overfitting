
from pathlib import Path

import pandas as pd

from performance_metrics import evaluate_strategies


def main():

    # STEP 1: Identify the results directory
    results_dir = Path("results")

    # STEP 2: Load the saved return matrix
    strategy_matrix = pd.read_parquet(
        results_dir / "strategy_returns_no_signal.parquet"
    )

    print("Loaded strategy matrix:", strategy_matrix.shape)

    # STEP 3: Calculate metrics for all strategies
    print("Calculating performance metrics...")

    # STEP 3: Calculate performance for all 8,800 strategies
    performance = evaluate_strategies(
        strategy_matrix
    )

    # Confirm that every strategy was evaluated
    print("Strategies evaluated:", len(performance))

    # Display the first five rows
    print(performance.head())

    # STEP 4: Display the first few results
    print("\nPerformance results:")
    print(performance.head())

    # STEP 5: Save the performance table
    performance.to_csv(
        results_dir / "performance_no_signal.csv",
        index=False
    )

    print("\nPerformance metrics saved successfully!")

    
    # Rank strategies by annualized Sharpe Ratio
    top_strategies = performance.sort_values(
        by="sharpe",
        ascending=False
    ).head(10)

    print("\nTop 10 strategies by Sharpe Ratio:")
    print(top_strategies)


    # Load the original strategy configurations
    configurations = pd.read_csv(
        results_dir / "strategy_configurations.csv"
    )

    # Combine performance results with strategy parameters
    detailed_results = performance.merge(
        configurations,
        on="strategy_id",
        how="left"
    )

    # Find the top 10 strategies by Sharpe Ratio
    top_detailed = detailed_results.sort_values(
        by="sharpe",
        ascending=False
    ).head(10)

    print("\nTop strategies with trading parameters:")

    print(
        top_detailed[
            [
                "strategy_id",
                "entry_day",
                "holding_period",
                "stop_loss",
                "side",
                "sharpe",
                "max_drawdown",
                "psr"
            ]
        ].to_string(index=False)
    )




if __name__ == "__main__":
    main()
