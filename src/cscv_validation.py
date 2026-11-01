import numpy as np
import pandas as pd
from cscv import split_blocks, generate_combinations

def create_time_blocks(strategy_matrix, S=8):
    """
    Divide the strategy-return matrix into S
    non-overlapping, contiguous time blocks.

    Parameters
    ----------
    strategy_matrix : pd.DataFrame
        T x N matrix of daily strategy returns.
    S : int
        Number of time blocks (must be even).

    Returns
    -------
    list
        List of arrays containing row indices
        for each time block.
    """

    row_indices = np.arange(len(strategy_matrix))[:, None]
    return [block[:, 0] for block in split_blocks(row_indices, S)]


################################################################################################################################################


def generate_cscv_combinations(blocks):
    """
    Generate every symmetric IS/OOS combination.

    Parameters
    ----------
    blocks : list
        List of contiguous time-block indices.

    Returns
    -------
    list
        Pairs of IS and OOS row indices.
    """

    # STEP 1: Determine the number of blocks
    S = len(blocks)

    # STEP 3: Store the IS/OOS splits
    splits = []

    for is_blocks, oos_blocks in generate_combinations(S):

        # Combine the row indices in chronological order
        is_indices = np.sort(
            np.concatenate([blocks[i] for i in is_blocks])
        )

        oos_indices = np.sort(
            np.concatenate([blocks[i] for i in oos_blocks])
        )

        # Save both sets of indices
        splits.append((is_indices, oos_indices))

    return splits

#################################################################################################################################################


def calculate_column_sharpes(return_matrix):
    """
    Legacy finite daily-return Sharpe; not the masked P&L experiment scorer.

    Each column represents one strategy.
    """

    # Calculate the mean return of each strategy
    mean_returns = return_matrix.mean(axis=0)

    # Calculate standard deviation for each strategy
    volatility = return_matrix.std(axis=0, ddof=1)

    # Calculate annualized Sharpe Ratios
    sharpes = (
        mean_returns / volatility
    ) * np.sqrt(252)

    # Replace infinite values with missing values
    sharpes = sharpes.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return sharpes


##############################################################################################################################################################################



from math import comb


def validate_effect_of_S(strategy_matrix, S_values=None):
    """
    Examine how changing S affects CSCV partitioning
    and the number of possible IS/OOS combinations.

    Does not calculate PBO. The paper's binomial SE bound is reported as
    a heuristic, not a calibrated standard error for dependent CSCV splits.
    """

    if S_values is None:
        S_values = [s for s in [4, 6, 8, 10, 12, 16, 20]
                    if s <= len(strategy_matrix) and len(strategy_matrix) % s == 0]

    # Number of observations in the return matrix
    T = len(strategy_matrix)

    # Store results for each tested value of S
    results = []

    for S in S_values:

        # Divide observations into S contiguous blocks
        blocks = create_time_blocks(strategy_matrix, S=S)

        # Count all possible symmetric IS/OOS combinations
        n_combinations = comb(S, S // 2)

        # Record the size of the resulting time blocks
        block_sizes = [len(block) for block in blocks]

        # Save statistics for this value of S
        results.append({
            "S": S,
            "T": T,
            "min_block_size": min(block_sizes),
            "max_block_size": max(block_sizes),
            "n_combinations": n_combinations,
            "is_observations": T // 2,
            "oos_observations": T // 2,
            "paper_binomial_se_upper_bound": np.sqrt(0.25 / n_combinations),
        })

    return pd.DataFrame(results)


#######################################################################################################################


def validate_effect_of_N(strategy_matrix, N_values=None):
    """
    Examine how the number of strategies (N)
    affects the resolution of relative OOS ranks.

    Section 4's theoretical ranks assume no ties. This does not measure
    observed ranks or PBO sensitivity to the chosen strategy universe.
    """

    # Total available strategies
    total_strategies = strategy_matrix.shape[1]
    if total_strategies < 2:
        raise ValueError("At least two strategies are required.")
    if N_values is None:
        N_values = sorted({n for n in [10, 50, 100, 500, 1000, total_strategies]
                           if 2 <= n <= total_strategies})

    # Store the results
    results = []

    for N in N_values:

        # Check that N is valid
        if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 2 or N > total_strategies:
            raise ValueError(
                f"N must be an integer between 2 and {total_strategies}"
            )

        # Select N strategies from the return matrix
        subset = strategy_matrix.iloc[:, :N]

        # Compute the spacing between adjacent relative ranks
        rank_resolution = 1 / (N + 1)

        # Calculate the possible relative rank values
        relative_ranks = (
            np.arange(1, N + 1) / (N + 1)
        )

        # Convert ranks into logits
        logits = np.log(
            relative_ranks / (1 - relative_ranks)
        )

        # Store characteristics of this N
        results.append({
            "N": N,
            "columns": subset.shape[1],
            "rank_resolution": rank_resolution,
            "min_logit": logits.min(),
            "max_logit": logits.max(),
            "unique_logit_values": len(np.unique(logits))
        })

    return pd.DataFrame(results)


######################################################################################################################


def validate_temporal_dependence(strategy_matrix, S=8):
    """
    Verify that CSCV preserves chronological order
    within each contiguous time block.

    This checks block integrity and boundary effects,
    not whether all temporal dependence is preserved.
    """

    # STEP 1: Divide the matrix into time blocks
    blocks = create_time_blocks(strategy_matrix, S=S)

    # STEP 2: Generate every IS/OOS combination
    splits = generate_cscv_combinations(blocks)
    training_sets = {is_idx.tobytes() for is_idx, _ in splits}

    results = []

    # STEP 3: Check each IS/OOS split
    for split_id, (is_idx, oos_idx) in enumerate(splits):

        # Indices should remain chronologically ordered
        is_ordered = np.all(np.diff(is_idx) > 0)
        oos_ordered = np.all(np.diff(oos_idx) > 0)

        # Count jumps between nonconsecutive observations
        is_gaps = np.sum(np.diff(is_idx) > 1)
        oos_gaps = np.sum(np.diff(oos_idx) > 1)

        # Check that IS and OOS have no overlap
        no_overlap = (
            len(np.intersect1d(is_idx, oos_idx)) == 0
        )

        # Verify every original row is assigned
        complete_split = (
            np.array_equal(np.sort(np.r_[is_idx, oos_idx]), np.arange(len(strategy_matrix)))
        )

        results.append({
            "split": split_id,
            "is_chronological": is_ordered,
            "oos_chronological": oos_ordered,
            "is_boundary_gaps": int(is_gaps),
            "oos_boundary_gaps": int(oos_gaps),
            "no_overlap": no_overlap,
            "complete_split": complete_split,
            "equal_sample_size": len(is_idx) == len(oos_idx),
            "oos_reused_as_is": oos_idx.tobytes() in training_sets,
        })

    return pd.DataFrame(results)


#####################################################################################################################


def validate_autocorrelation_preservation(
    T=1000,
    S=8,
    phi=0.7,
    seed=42
):
    """
    Custom AR(1) demonstration, not an experiment specified in the main PDF.
    Measures only genuine adjacent pairs, not correlations across block gaps.
    """

    if not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("Stationary AR(1) requires finite abs(phi) < 1.")
    # STEP 1: Generate reproducible random shocks
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 1, T + 200)

    # STEP 2: Generate AR(1) observations
    series = np.zeros(T + 200)

    for t in range(1, len(series)):
        series[t] = phi * series[t - 1] + shocks[t]

    # Remove burn-in observations
    series = pd.Series(series[200:])

    # STEP 3: Create CSCV blocks
    blocks = create_time_blocks(
        pd.DataFrame({"returns": series}),
        S=S
    )

    splits = generate_cscv_combinations(blocks)

    # STEP 4: Calculate original autocorrelation
    original_acf = series.autocorr(lag=1)

    # STEP 5: Calculate autocorrelation in each CSCV IS sample
    cscv_acfs = []

    for is_idx, _ in splits:
        # Exclude artificial adjacent pairs across block gaps
        left_idx = is_idx[:-1]
        right_idx = is_idx[1:]
        consecutive = (right_idx - left_idx) == 1

        x = series.iloc[left_idx[consecutive]].to_numpy()
        y = series.iloc[right_idx[consecutive]].to_numpy()
        if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
            raise ValueError("Each IS split needs at least two nonconstant adjacent pairs for ACF.")

        cscv_acfs.append(np.corrcoef(x, y)[0, 1])

    # STEP 6: Shuffle individual observations
    shuffled = pd.Series(rng.permutation(series.to_numpy()))
    shuffled_acf = shuffled.autocorr(lag=1)

    return pd.DataFrame([{
        "original_acf": original_acf,
        "mean_cscv_acf": np.mean(cscv_acfs),
        "shuffled_acf": shuffled_acf,
        "true_phi": phi,
        "S": S
    }])


#############################################################################


def validate_pbo_accuracy(estimated_pbo, reference_pbo):
    """
    Evaluate CSCV PBO estimation accuracy against
    reference overfitting probabilities.

    Parameters
    ----------
    estimated_pbo : array-like
        CSCV PBO estimates from the main implementation.
    reference_pbo : array-like
        Reference probabilities from independent experiments.

    Returns
    -------
    dict
        Bias, MAE, and RMSE.
    """

    # STEP 1: Convert the inputs into NumPy arrays
    estimated = np.asarray(estimated_pbo, dtype=float)
    reference = np.asarray(reference_pbo, dtype=float)

    # STEP 2: Validate the inputs
    if estimated.shape != reference.shape:
        raise ValueError("The arrays must have the same shape.")

    if estimated.size == 0:
        raise ValueError("At least one comparison is required.")

    if not np.isfinite(estimated).all() or not np.isfinite(reference).all():
        raise ValueError("PBO values must be finite.")

    if np.any((estimated < 0) | (estimated > 1)):
        raise ValueError("Estimated PBO must be between 0 and 1.")

    if np.any((reference < 0) | (reference > 1)):
        raise ValueError("Reference PBO must be between 0 and 1.")

    # STEP 3: Calculate estimation errors
    errors = estimated - reference

    # STEP 4: Compute summary statistics
    bias = np.mean(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    return {
        "bias": float(bias),
        "mae": float(mae),
        "rmse": float(rmse),
        "n_scenarios": int(estimated.size)
    }


################################################################################


def validate_model_free(T=1000, N=100, S=8, seed=42):
    """
    Custom sampling demonstration, not the paper's model-free validation.

    This tests input compatibility and sampling
    invariance, not PBO estimation accuracy.
    """

    # STEP 1: Initialize reproducible random generator
    rng = np.random.default_rng(seed)

    # STEP 2: Generate three different return matrices
    datasets = {
        "Normal": rng.normal(
            loc=0.0,
            scale=0.01,
            size=(T, N)
        ),

        "Student_t": rng.standard_t(
            df=5,
            size=(T, N)
        ) * 0.01,

        "Skewed": (
            rng.exponential(
                scale=0.01,
                size=(T, N)
            ) - 0.01
        )
    }

    results = []

    # STEP 3: Apply identical CSCV sampling to each dataset
    for distribution, returns in datasets.items():

        # Convert returns into a DataFrame
        matrix = pd.DataFrame(returns)

        # Create contiguous blocks
        blocks = create_time_blocks(matrix, S=S)

        # Generate symmetric IS/OOS combinations
        splits = generate_cscv_combinations(blocks)

        # Check the first split
        is_idx, oos_idx = splits[0]

        # Confirm the structure is valid
        no_overlap = (
            len(np.intersect1d(is_idx, oos_idx)) == 0
        )

        complete_split = (
            len(is_idx) + len(oos_idx) == T
        )

        results.append({
            "distribution": distribution,
            "T": matrix.shape[0],
            "N": matrix.shape[1],
            "S": S,
            "n_splits": len(splits),
            "no_overlap": no_overlap,
            "complete_split": complete_split
        })

    return pd.DataFrame(results)


####################################################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Section 4 structural diagnostics, not PBO accuracy tests.")
    parser.add_argument("returns", help="Path to a saved strategy-return Parquet matrix")
    parser.add_argument("--S", type=int, default=8)
    args = parser.parse_args()
    matrix = pd.read_parquet(args.returns)
    print("Effect of S (theoretical):")
    print(validate_effect_of_S(matrix).to_string(index=False))
    print("Effect of N (theoretical, no ties):")
    print(validate_effect_of_N(matrix).to_string(index=False))
    print("Chronology and symmetry:")
    print(validate_temporal_dependence(matrix, S=args.S).to_string(index=False, max_rows=10))
