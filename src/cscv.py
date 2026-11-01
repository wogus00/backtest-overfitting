import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import combinations
from scipy.stats import rankdata
from performance_metrics import column_sharpe_ratios, column_pnl_sharpe_ratios


def split_blocks(M, S=16):
    """Divide the T x N strategy-return matrix into S blocks."""
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[1] == 0:
        raise ValueError("M must be a T x N matrix with at least one column.")
    if isinstance(S, (bool, np.bool_)) or not isinstance(S, (int, np.integer)) or S < 2 or S % 2:
        raise ValueError("S must be an even integer >= 2.")
    T, N = M.shape
    if T < S or T % S:
        raise ValueError("T must be divisible by S, with no empty blocks.")
    return list(np.split(M, S, axis=0))


def generate_combinations(S=16):
    """S/2 IS blocks."""
    results = []
    block_indices = range(S)

    for in_sample_indices in combinations(block_indices, S // 2):
        in_sample_set = set(in_sample_indices)

        out_of_sample_indices = []

        for i in block_indices:
            if i not in in_sample_set:
                out_of_sample_indices.append(i)

        results.append((in_sample_indices, out_of_sample_indices))

    return results

def compute_relative_rank(out_of_sample_performance, selected_idx):
    """Rank IS strategy among OOS strategies."""
    out_of_sample_performance = np.asarray(out_of_sample_performance, dtype=float)
    if out_of_sample_performance.ndim != 1 or out_of_sample_performance.size < 2 or not np.isfinite(out_of_sample_performance).all():
        raise ValueError("OOS scores must be a finite vector with at least two strategies.")

    ranks = rankdata(out_of_sample_performance, method="average")
    out_of_sample_rank = float(ranks[selected_idx])
    N = len(out_of_sample_performance)
    relative_rank = out_of_sample_rank / (N + 1)

    return out_of_sample_rank, relative_rank



def compute_logit(relative_rank):
    if not 0 < relative_rank < 1:
        raise ValueError("Relative rank must be strictly between zero and one.")
    return float(np.log(relative_rank / (1 - relative_rank)))



def cscv(M, S=16, metric=None, calendar_days=None):
    """
    Run CSCV using a specified performance metric.

    calendar_days supplies per-row exposure weights for masked price-unit P&L.
    In that mode, one fixed universe is retained with defined Sharpe in every
    split; exclusions are reported in result.attrs['eligibility'].
    Otherwise all metrics must return an array of length N,
    where higher values indicate better performance.
        * for maximum drawdown, return negative values so that higher is better
    """
    M = np.asarray(M, dtype=float)

    blocks = split_blocks(M, S)
    is_pnl = calendar_days is not None
    combinations_ = generate_combinations(S)
    original_indices = np.arange(M.shape[1])
    eligibility = None
    if is_pnl:
        if metric is not None:
            raise ValueError("Custom metrics are not supported in P&L mode.")
        weights = np.asarray(calendar_days, dtype=float)
        if weights.shape != (len(M),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Expected one finite, positive calendar weight per row.")
        if np.isinf(M).any():
            raise ValueError("P&L may contain NaN for inactivity, but not infinity.")
        day_blocks = np.split(weights, S)
        counts = np.array([np.sum(~np.isnan(block), axis=0) for block in blocks])
        minima = np.array([np.min(np.where(np.isnan(block), np.inf, block), axis=0) for block in blocks])
        maxima = np.array([np.max(np.where(np.isnan(block), -np.inf, block), axis=0) for block in blocks])
        enough = np.sort(counts, axis=0)[:S // 2].sum(axis=0) > 2
        varied = np.ones(M.shape[1], dtype=bool)
        for chosen, _ in combinations_:
            varied &= maxima[list(chosen)].max(axis=0) > minima[list(chosen)].min(axis=0)
        eligible = enough & varied
        eligibility = pd.DataFrame({"column_index": original_indices,
                                    "eligible": eligible,
                                    "enough_observations_every_split": enough,
                                    "nonconstant_every_split": varied})
        original_indices = original_indices[eligible]
        blocks = [block[:, eligible] for block in blocks]
        if len(original_indices) < 2:
            raise ValueError("Fewer than two strategies have defined P&L Sharpe in every split.")
        excluded = int((~eligible).sum())
        if excluded:
            tqdm.write(f"CSCV eligibility: excluded {excluded}/{M.shape[1]} strategies; fixed N={len(original_indices)}.")
    elif not np.isfinite(M).all():
        raise ValueError("M must contain only finite returns.")
    metric = column_sharpe_ratios if metric is None else metric
    if not callable(metric):
        raise TypeError("metric must be callable and return one score per strategy.")
    N = len(original_indices)
    if N < 2:
        raise ValueError("CSCV requires at least two strategies.")
    results = []

    for split_id, (in_sample_indices, out_of_sample_indices) in enumerate(tqdm(combinations_, desc="CSCV", unit="split")):
        IS = np.vstack([blocks[i] for i in in_sample_indices])
        OOS = np.vstack([blocks[i] for i in out_of_sample_indices])

        # Evaluate all strategies
        if is_pnl:
            is_days = sum(day_blocks[i].sum() for i in in_sample_indices)
            oos_days = sum(day_blocks[i].sum() for i in out_of_sample_indices)
            is_performance = column_pnl_sharpe_ratios(IS, is_days)
            oos_performance = column_pnl_sharpe_ratios(OOS, oos_days)
        else:
            is_performance = np.asarray(metric(IS), dtype=float)
            oos_performance = np.asarray(metric(OOS), dtype=float)
        for sample, scores in (("IS", is_performance), ("OOS", oos_performance)):
            if scores.shape != (N,) or not np.isfinite(scores).all():
                raise ValueError(
                    f"Split {split_id} {sample}: metric must return {N} finite scores; "
                    "check for no-trade or zero-volatility strategies."
                )

        # Select best IS strategy
        selected_idx = np.argmax(is_performance)

        selected_is_performance = is_performance[selected_idx]
        selected_oos_performance = oos_performance[selected_idx]

        # Rank selected strategy OOS
        oos_rank, relative_rank = compute_relative_rank(oos_performance, selected_idx)

        logit = compute_logit(relative_rank)

        # Get daily OOS returns of the selected strategy
        selected_oos_returns = OOS[:, selected_idx]

        # Calculate total OOS return (assuming simple returns)
        total = np.nansum(selected_oos_returns) if is_pnl else np.prod(1 + selected_oos_returns) - 1

        results.append({"split_id": split_id,
                        "selected_idx": original_indices[selected_idx],
                        "performance_is": selected_is_performance,
                        "performance_oos": selected_oos_performance,
                        "oos_rank": oos_rank,
                        "relative_rank": relative_rank,
                        "logit": logit,
                        "benchmark_oos": np.mean(oos_performance),
                        "oos_total_pnl" if is_pnl else "oos_total_return": total})
        if is_pnl:
            results[-1].update(is_calendar_days=is_days, oos_calendar_days=oos_days,
                               n_strategies=N,
                               is_recorded_observations=int(np.count_nonzero(~np.isnan(IS[:, selected_idx]))),
                               oos_recorded_observations=int(np.count_nonzero(~np.isnan(selected_oos_returns))))
    result = pd.DataFrame(results)
    if eligibility is not None:
        result.attrs["eligibility"] = eligibility
    return result
