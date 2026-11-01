"""Plot saved Sharpe experiments, replaying OOS scoring for pooled dominance.

Run: python -m backtest_overfitting.plot --experiment paper_source_S8 --seed 1
Requires numpy, pandas, pyarrow, and matplotlib.
"""
import argparse
import math

import numpy as np
import pandas as pd

from . import paths


def require_finite(frame, columns, label):
    if frame.empty or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: expected nonempty, finite {columns}")


def pooled_dominance(returns, saved_cscv, S, is_pnl):
    """Pool individual scores across every OOS split of the saved universe."""
    from .cscv import split_blocks, generate_combinations
    from .performance_metrics import (calendar_day_weights, column_pnl_sharpe_ratios,
                                     column_sharpe_ratios)
    from .overfit_statistics import stochastic_dominance
    from tqdm import tqdm

    blocks = split_blocks(returns.to_numpy(dtype=float), S)
    combinations = generate_combinations(S)
    if len(saved_cscv) != len(combinations) or not np.array_equal(saved_cscv.split_id, np.arange(len(combinations))):
        raise ValueError("Saved CSCV rows must contain every split in generation order.")
    if not returns.columns.is_unique or not saved_cscv.strategy_id.isin(returns.columns).all():
        raise ValueError("Saved winners must belong to the fixed strategy universe.")
    winners = returns.columns.get_indexer(saved_cscv.strategy_id)
    days = np.split(calendar_day_weights(returns.index), S) if is_pnl else None
    # shortcut: pooled scores live in RAM; use disk-backed sorting if large S exceeds memory.
    pooled = np.empty((len(combinations), returns.shape[1]))
    for split, (_, oos) in enumerate(tqdm(combinations, desc="Pooling OOS scores", unit="split")):
        sample = np.vstack([blocks[i] for i in oos])
        scores = (column_pnl_sharpe_ratios(sample, sum(days[i].sum() for i in oos))
                  if is_pnl else column_sharpe_ratios(sample))
        if not np.isfinite(scores).all():
            raise ValueError(f"Split {split}: undefined OOS scores in the saved universe.")
        if (not np.isclose(scores[winners[split]], saved_cscv.performance_oos.iloc[split], rtol=1e-10, atol=1e-12)
                or not np.isclose(scores.mean(), saved_cscv.benchmark_oos.iloc[split], rtol=1e-10, atol=1e-12)):
            raise ValueError(f"Split {split}: replayed OOS scores disagree with the saved experiment.")
        pooled[split] = scores
    result = stochastic_dominance(saved_cscv, benchmark_scores=pooled.ravel())
    curve = pd.DataFrame({"score": result["x"], "cdf_selected": result["cdf_selected"],
                          "cdf_pooled": result["cdf_benchmark"],
                          "integrated_cdf_difference": result["sd2"]})
    summary = {"benchmark": "pooled_oos_strategy_scores", "selected_count": len(saved_cscv),
               "benchmark_count": pooled.size,
               "first_order_dominance_vs_pooled": result["first_order_dominance"],
               "second_order_dominance_vs_pooled": result["second_order_dominance"]}
    return curve, summary


def load_dataset(experiment, seed, name):
    scores = paths.tables_dir(experiment, seed) / name / "sharpe"
    cscv = pd.read_parquet(scores / "cscv.parquet")
    performance = pd.read_parquet(scores / "performance.parquet")
    summary = pd.read_csv(scores / "summary.csv").iloc[0]
    require_finite(cscv, ["performance_is", "performance_oos", "logit"], name)
    candidates = performance.loc[np.isfinite(performance["sharpe"])]
    if candidates.empty:
        raise ValueError(f"{name}: no finite full-sample Sharpe ratios")
    best = candidates.loc[candidates["sharpe"].idxmax()]
    returns = pd.read_parquet(paths.processed_dir(experiment, seed) / "returns" / f"{name}.parquet")
    sid = best["strategy_id"]
    if sid not in returns.columns:
        raise ValueError(f"{name}: strategy ID {sid} missing from returns")
    price_frame = pd.read_parquet(
        paths.data_dir(experiment, seed) / f"{name}.parquet"
    )
    prices = price_frame["price"]
    is_pnl = "oos_total_pnl" in cscv
    if len(prices) != len(returns) or not returns.index.is_unique:
        raise ValueError(f"{name}: prices and return dates do not align")
    if is_pnl:
        from .performance_metrics import pnl_sharpe_ratio
        if "date" not in price_frame or not returns.index.equals(pd.DatetimeIndex(price_frame["date"])):
            raise ValueError(f"{name}: stored price dates do not match P&L dates")
        require_finite(returns[[sid]].dropna(), [sid], name)
        expected = pnl_sharpe_ratio(returns[sid], returns.index[0], returns.index[-1])
        if not np.isclose(best["sharpe"], expected):
            raise ValueError(f"{name}: full-sample Sharpe disagrees with recorded P&L")
        eligibility = pd.read_parquet(scores / "eligibility.parquet")
        ids = eligibility.loc[eligibility.eligible, "strategy_id"]
        if set(performance.strategy_id) != set(ids) or not cscv.strategy_id.isin(ids).all():
            raise ValueError(f"{name}: inconsistent eligible strategy universe")
        if not cscv.n_strategies.eq(len(ids)).all():
            raise ValueError(f"{name}: inconsistent CSCV strategy count")
    else:
        require_finite(returns, [sid], name)
        ids = returns.columns
    require_finite(prices.to_frame(), ["price"], name)
    splits = [s for s in range(2, 34, 2) if math.comb(s, s // 2) == len(cscv)]
    if len(splits) != 1 or len(returns) % splits[0]:
        raise ValueError(f"{name}: invalid equal-block CSCV split count")
    pbo = (cscv["logit"] < 0).mean()
    loss = (cscv["performance_oos"] < 0).mean()
    if not np.isclose(pbo, summary["pbo"]):
        raise ValueError(f"{name}: saved PBO disagrees with logits")
    if not np.isclose(loss, summary["probability_below_oos_benchmark"]) or summary["oos_score_benchmark"] != 0:
        raise ValueError(f"{name}: saved negative-Sharpe probability disagrees")
    if "probability_of_loss" in summary and not np.isclose(loss, summary["probability_of_loss"]):
        raise ValueError(f"{name}: paper loss probability disagrees with OOS scores")
    dominance, dominance_summary = pooled_dominance(returns[ids], cscv, splits[0], is_pnl)
    return dict(cscv=cscv, dominance=dominance, summary=summary, best=best,
                dominance_summary=dominance_summary,
                returns=returns[sid], prices=prices.to_numpy(),
                T=len(returns), N=int(cscv.n_strategies.iloc[0]) if is_pnl else returns.shape[1],
                S=splits[0], is_pnl=is_pnl)


# The paper uses Times for chart text and Helvetica for figure captions.
PAPER_STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.linewidth": .45, "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.width": .45,
    "ytick.major.width": .45, "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": .55, "legend.fontsize": 7, "legend.framealpha": 1,
    "legend.fancybox": False, "legend.edgecolor": "black",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}
BOX = dict(boxstyle="round,pad=0.25,rounding_size=0.08", facecolor="white",
           edgecolor="black", linewidth=.45)


def plot_backtest(ax, data):
    import matplotlib.dates as mdates

    returns = data["returns"]
    is_pnl = data.get("is_pnl", False)
    cumulative = returns.fillna(0.).cumsum() if is_pnl else 100 * ((1 + returns).cumprod() - 1)
    ax.plot(returns.index, cumulative,
            color="blue", label="Strategy")
    prices_ax = ax.twinx()
    underlying = data["prices"] if is_pnl else data["prices"] - data["prices"][0]
    prices_ax.plot(returns.index, underlying,
                   color="green", linestyle=":", linewidth=.5, label="Underlying")
    prices_ax.set_ylabel("Prices")
    prices_ax.tick_params(left=False)
    # Keep the twin-axis price curve behind the callout and legend.
    ax.set_zorder(prices_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("Cumulative P&L (price units)" if is_pnl else "Performance (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 9]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelrotation=90, pad=-8)
    ax.margins(x=.02)
    # PSR in the paper's callout is a z statistic, not the saved PSR probability.
    observations = returns.dropna()
    if is_pnl:
        from scipy.stats import skew, kurtosis
        sr = observations.mean() / observations.std(ddof=0)
        variance = 1 - skew(observations) * sr + (kurtosis(observations, fisher=False) - 1) * sr**2 / 4
    else:
        sr = returns.mean() / returns.std(ddof=1)
        variance = 1 - returns.skew() * sr + (returns.kurt() + 2) * sr**2 / 4
    psr_stat = sr * np.sqrt(len(observations) - 1) / np.sqrt(variance) if variance > 0 else np.nan
    annotation = f"SR={data['best']['sharpe']:.2f} PSR={psr_stat:.2f}"
    if is_pnl:
        frequency = len(observations) * 365.25 / ((returns.index[-1] - returns.index[0]).days + 1)
        annotation += f" Freq={frequency:.2f}"
    ax.text(.04, .94, annotation, transform=ax.transAxes, va="top", bbox=BOX)
    lines = ax.get_lines() + prices_ax.get_lines()
    legend = ax.legend(lines, [line.get_label() for line in lines], loc="center left")
    legend.get_frame().set_linewidth(.45)
    return prices_ax


def plot_degradation(ax, data):
    c = data["cscv"]
    x, y = c["performance_is"], c["performance_oos"]
    ax.scatter(x, y, s=5, facecolors="red", edgecolors="black", linewidths=.25)
    summary = data["summary"]
    if len(x) > 2 and np.ptp(x) > 0:
        line = np.array([x.min(), x.max()])
        ax.plot(line, summary["degradation_intercept"] + summary["degradation_slope"] * line,
                color="black", linewidth=.5)
        adjusted_r2 = 1 - (1 - summary["degradation_r2"]) * (len(x) - 1) / (len(x) - 2)
        regression = (f"[SR OOS]={summary['degradation_intercept']:.2f}"
                      f"{summary['degradation_slope']:+.2f}*[SR IS]+err | adjR²={adjusted_r2:.2f}")
    else:
        regression = "Regression undefined (constant or insufficient IS scores)"
    ax.text(.04, .94, regression, transform=ax.transAxes, va="top", fontsize=9, bbox=BOX)
    ax.text(.04, .04, f"Prob[SR OOS<0]={(y < 0).mean():.2f}",
            transform=ax.transAxes, va="bottom", bbox=BOX)
    ax.set(xlabel="SR IS", ylabel="SR OOS", title=f"{data['S']} | OOS Perf. Degradation")
    ax.margins(x=.06, y=.18)


def plot_logits(ax, data):
    from matplotlib.ticker import FormatStrFormatter

    values = data["cscv"]["logit"].to_numpy()
    # Match the paper's narrow bars; do not smooth or resample observations.
    extent = max(6, 2 * math.ceil(np.abs(values).max() / 2))
    bins = np.linspace(-extent, extent, 51)
    heights, _, _ = ax.hist(values, bins=bins, density=True, color="#459b45",
                            edgecolor="black", linewidth=.3)
    std = values.std(ddof=0)
    peak = heights.max()
    if std > 0:
        x = np.linspace(-extent, extent, 600)
        density = np.exp(-.5 * ((x - values.mean()) / std)**2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, density, color="red", linestyle=":", linewidth=.6)
        peak = max(peak, density.max())
    ax.text(.04, .94, f"Prob Overfit={data['summary']['pbo']:.2f}",
            transform=ax.transAxes, va="top", bbox=BOX)
    ax.set(xlabel="Logits", ylabel="Frequency", title=f"{data['S']} | Hist. of Rank Logits",
           xlim=(-extent, extent), ylim=(0, peak * 1.12))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def plot_dominance(ax, data):
    from matplotlib.ticker import FormatStrFormatter

    d = data["dominance"].sort_values("score")
    x = d["score"].to_numpy()
    ax.step(x, d["cdf_selected"], where="post", color="blue", linestyle=":", label="optimized")
    ax.step(x, d["cdf_pooled"], where="post", color="red", linestyle="--", label="non-optimized (pooled)")
    area_ax = ax.twinx()
    area_ax.plot(x, d["integrated_cdf_difference"], color="green", linestyle=":", label="SD2")
    area_ax.set_ylabel("2nd Order Stoch. Dominance")
    area_ax.tick_params(left=False)
    ax.set_zorder(area_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    area_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set(xlabel="SR optimized vs. non-optimized", ylabel="Frequency",
           title=f"{data['S']} | OOS Cumul. Dist.", ylim=(0, 1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    for axis, position in [(ax, "center left"), (area_ax, "center right")]:
        axis.legend(loc=position).get_frame().set_linewidth(.45)
    return area_ax


def paper_frame(fig, number, caption):
    from matplotlib.lines import Line2D
    import textwrap

    fig.text(.045, .955, f"FIGURE {number}", family="sans-serif", fontsize=9, weight="bold", va="top")
    fig.text(.175, .955, textwrap.fill(caption, width=80), family="sans-serif", fontsize=9, va="top")
    for y in (.865, .035):
        fig.add_artist(Line2D([.045, .955], [y, y], transform=fig.transFigure,
                              color="black", linewidth=1.25))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="paper_source_S8")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    plot_experiment(args.experiment, args.seed)


def plot_experiment(experiment, seed):
    datasets = [load_dataset(experiment, seed, name)
                for name in ("no_signal", "seasonal")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = paths.figures_dir(experiment, seed)
    output.mkdir(parents=True, exist_ok=True)
    tables = paths.tables_dir(experiment, seed)
    tables.mkdir(parents=True, exist_ok=True)
    for label, data in zip(("no_signal", "seasonal"), datasets):
        data["dominance"].to_parquet(tables / f"{label}_pooled_dominance.parquet", index=False)
    pd.DataFrame([{"scenario": label, **data["dominance_summary"]}
                  for label, data in zip(("no_signal", "seasonal"), datasets)]).to_csv(
                      tables / "pooled_dominance_summary.csv", index=False)
    plots = [("backtest", plot_backtest), ("degradation", plot_degradation),
             ("logits", plot_logits), ("dominance", plot_dominance)]
    with plt.rc_context(PAPER_STYLE):
        # Individual exports reproduce the paper's figure layout and caption rules.
        for example, (label, data) in enumerate(zip(["no_signal", "seasonal"], datasets), 1):
            for offset, (name, draw) in enumerate(plots):
                number = 6 + (example - 1) * 4 + offset
                fig = plt.figure(figsize=(7, 5.6))
                ax = fig.add_axes([.19, .18, .62, .56])
                draw(ax, data)
                if name == "backtest":
                    caption = f"Backtested performance of a seasonal strategy (example {example})."
                else:
                    ending = {"degradation": "performance degradation" if example == 1 else "monthly effect",
                              "logits": "logit distribution",
                              "dominance": "stochastic dominance"}[name]
                    caption = (f"CSCV analysis of the backtest of a seasonal strategy (example {example}): "
                               f"{ending}.")
                paper_frame(fig, number, caption)
                fig.savefig(output / f"{label}_{name}.png", dpi=250)
                plt.close(fig)
        # Retain the convenient overview, using the same four paper-style panels.
        fig, axes = plt.subplots(4, 2, figsize=(14, 17))
        fig.subplots_adjust(left=.08, right=.91, bottom=.055, top=.94, hspace=.45, wspace=.5)
        for col, data in enumerate(datasets):
            for row, (_, draw) in enumerate(plots):
                draw(axes[row, col], data)
        fig.text(.28, .98, "No signal", ha="center", va="top", fontsize=13)
        fig.text(.73, .98, "Seasonal signal", ha="center", va="top", fontsize=13)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"comparison.{suffix}", dpi=200)
        plt.close(fig)
    print(f"Saved individual PNG figures and comparison.png/comparison.pdf to {output}")


if __name__ == "__main__":
    main()
