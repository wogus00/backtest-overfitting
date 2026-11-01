import argparse
from . import paths

import pandas as pd

from .main import measure_performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="paper_source_S8")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", choices=["no_signal", "seasonal"], default="seasonal")
    args = parser.parse_args()
    processed = paths.processed_dir(args.experiment, args.seed)
    output = paths.tables_dir(args.experiment, args.seed) / args.dataset / "sharpe"
    if not (output / "eligibility.parquet").exists():
        raise ValueError("Run the P&L pipeline first; legacy percentage-return results are not supported here.")
    pnl = pd.read_parquet(processed / "returns" / f"{args.dataset}.parquet")
    configurations = pd.read_parquet(processed / "strategy_configurations.parquet")
    eligibility = pd.read_parquet(output / "eligibility.parquet")
    ids = eligibility.loc[eligibility.eligible, "strategy_id"].tolist()
    performance = measure_performance(pnl[ids], configurations)
    performance.to_csv(output / "performance.csv", index=False)
    print(performance.sort_values("sharpe", ascending=False).head(10).to_string(index=False))
    print(f"Saved {len(performance)} eligible strategies to {output / 'performance.csv'}")


if __name__ == "__main__":
    main()
