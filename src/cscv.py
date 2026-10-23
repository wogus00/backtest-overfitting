import numpy as np
import pandas as pd

from itertools import combinations
from scipy.stats import rankdata
from performance_metrics import sharpe_ratio


def split_blocks(M, S=16):
    """Divide the T x N strategy-return matrix into S blocks."""
    M = np.asarray(M)
    T, N = M.shape
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

    ranks = rankdata(out_of_sample_performance, method="average")
    out_of_sample_rank = float(ranks[selected_idx])
    N = len(out_of_sample_performance)
    relative_rank = out_of_sample_rank / (N + 1)

    return out_of_sample_rank, relative_rank



def compute_logit(relative_rank):
    return float(np.log(relative_rank / (1 - relative_rank)))



def cscv(M, S=16, metric="sharpe_ratio"):
    """
    Run CSCV using a specified performance metric.

    All metrics must return an array of length N,
    where higher values indicate better performance.
        * for maximum drawdown, return negative values so that higher is better
    """
    M = np.asarray(M, dtype=float)

    T, N = M.shape

    blocks = split_blocks(M, S)
    results = []

    for split_id, (in_sample_indices, out_of_sample_indices) in enumerate(generate_combinations(S)):
        IS = np.vstack([blocks[i] for i in in_sample_indices])
        OOS = np.vstack([blocks[i] for i in out_of_sample_indices])

        # Evaluate all strategies
        is_performance = np.asarray(metric(IS), dtype=float)
        oos_performance = np.asarray(metric(OOS), dtype=float)


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
        oos_total_return = np.prod(1 + selected_oos_returns) - 1

        results.append({"split_id": split_id,
                        "selected_idx": selected_idx,
                        "performance_is": selected_is_performance,
                        "performance_oos": selected_oos_performance,
                        "oos_rank": oos_rank,
                        "relative_rank": relative_rank,
                        "logit": logit,
                        "benchmark_oos": np.mean(oos_performance),
                        "oos_total_return": oos_total_return})
    return pd.DataFrame(results)