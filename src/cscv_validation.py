import itertools   
import numpy as np
import pandas as pd

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

    # STEP 1: Check that S is a valid number
    if S < 2 or S % 2 != 0:
        raise ValueError("S must be an even integer >= 2.")

    # STEP 2: Determine the number of observations
    T = len(strategy_matrix)

    # STEP 3: Require enough rows for S nonempty blocks
    if T < S:
        raise ValueError("S cannot exceed the number of observations.")

    # STEP 4: Create consecutive row indices
    row_indices = np.arange(T)

    # STEP 5: Divide observations into S contiguous blocks
    blocks = np.array_split(row_indices, S)

    return blocks


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

    # Half of the blocks will be in-sample
    half = S // 2

    # STEP 2: Generate all ways to select S/2 blocks
    combinations = itertools.combinations(
        range(S),
        half
    )

    # STEP 3: Store the IS/OOS splits
    splits = []

    for is_blocks in combinations:

        # OOS blocks are the complement of IS blocks
        oos_blocks = [
            i for i in range(S)
            if i not in is_blocks
        ]

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
    Calculate annualized Sharpe Ratios for all strategies.

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

    Does not calculate PBO.
    """

    if S_values is None:
        S_values = [4, 6, 8, 10, 12, 16, 20]

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
            "n_combinations": n_combinations
        })

    return pd.DataFrame(results)


#######################################################################################################################


def validate_effect_of_N(strategy_matrix, N_values=None):
    """
    Examine how the number of strategies (N)
    affects the resolution of relative OOS ranks.

    Does not calculate PBO.
    """

    if N_values is None:
        N_values = [10, 50, 100, 500, 1000, 8800]

    # Total available strategies
    total_strategies = strategy_matrix.shape[1]

    # Store the results
    results = []

    for N in N_values:

        # Check that N is valid
        if N < 2 or N > total_strategies:
            raise ValueError(
                f"N must be between 2 and {total_strategies}"
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
            len(is_idx) + len(oos_idx)
            == len(strategy_matrix)
        )

        results.append({
            "split": split_id,
            "is_chronological": is_ordered,
            "oos_chronological": oos_ordered,
            "is_boundary_gaps": int(is_gaps),
            "oos_boundary_gaps": int(oos_gaps),
            "no_overlap": no_overlap,
            "complete_split": complete_split
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
    Compare lag-1 autocorrelation in an AR(1) series
    under original ordering, CSCV, and random shuffling.
    """

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
        selected = series.iloc[is_idx].reset_index(drop=True)

        # Exclude artificial adjacent pairs across block gaps
        left_idx = is_idx[:-1]
        right_idx = is_idx[1:]
        consecutive = (right_idx - left_idx) == 1

        x = series.iloc[left_idx[consecutive]].to_numpy()
        y = series.iloc[right_idx[consecutive]].to_numpy()

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
    Test whether the CSCV sampling framework works
    across different return distributions.

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

    # Load our previously saved strategy-return matrix
    strategy_matrix = pd.read_parquet(
        "results/strategy_returns_no_signal.parquet"
    )

    # Divide the data into eight time blocks
    blocks = create_time_blocks(
        strategy_matrix,
        S=8
    )

    print("Number of blocks:", len(blocks))

    for i, block in enumerate(blocks):
        print(
            f"Block {i + 1}: "
            f"{len(block)} observations, "
            f"rows {block[0]} to {block[-1]}"
        )

    # Ensure every observation is assigned exactly once
    combined_indices = np.concatenate(blocks)

    assert np.array_equal(
        combined_indices,
        np.arange(len(strategy_matrix))
    )

    print("All observations assigned correctly!")


    # Generate all symmetric combinations
    splits = generate_cscv_combinations(blocks)

    print("\nTotal CSCV combinations:", len(splits))

    # Inspect the first combination
    is_indices, oos_indices = splits[0]

    print("IS observations:", len(is_indices))
    print("OOS observations:", len(oos_indices))

    # Verify no observations overlap
    assert len(np.intersect1d(is_indices, oos_indices)) == 0

    # Verify each split includes all observations
    assert len(is_indices) + len(oos_indices) == len(strategy_matrix)

    # Verify the expected number of combinations
    assert len(splits) == 70

    print("CSCV combinations verified!")


    # STEP 3: Test the first CSCV combination

    # Extract row indices for the first split
    is_indices, oos_indices = splits[0]

    # Create the IS and OOS return matrices
    is_matrix = strategy_matrix.iloc[is_indices]
    oos_matrix = strategy_matrix.iloc[oos_indices]

    # Calculate Sharpe Ratios for all strategies
    is_sharpes = calculate_column_sharpes(is_matrix)
    oos_sharpes = calculate_column_sharpes(oos_matrix)

    # Find the strategy with the highest IS Sharpe Ratio
    best_strategy = is_sharpes.idxmax()

    # Get that strategy's IS and OOS performance
    best_is_sharpe = is_sharpes.loc[best_strategy]
    selected_oos_sharpe = oos_sharpes.loc[best_strategy]

    print("\nFirst CSCV split:")
    print("Best IS strategy:", best_strategy)
    print("Best IS Sharpe:", round(best_is_sharpe, 4))
    print("Same strategy OOS Sharpe:", round(selected_oos_sharpe, 4))


    # VALIDATION 1: Effect of S
    print("\n--- Effect of S ---")

    s_results = validate_effect_of_S(
        strategy_matrix
    )

    print(s_results.to_string(index=False))


    # VALIDATION 2: Effect of N
    print("\n--- Effect of N ---")

    n_results = validate_effect_of_N(
        strategy_matrix
    )

    print(n_results.to_string(index=False))


    # VALIDATION 3: Temporal dependence
    print("\n--- Temporal Dependence ---")

    temporal_results = validate_temporal_dependence(
        strategy_matrix,
        S=8
    )

    print(temporal_results.head(10).to_string(index=False))

    # Confirm all splits preserve chronological ordering
    assert temporal_results["is_chronological"].all()
    assert temporal_results["oos_chronological"].all()

    # Confirm all splits are disjoint and complete
    assert temporal_results["no_overlap"].all()
    assert temporal_results["complete_split"].all()

    print("\nAll 70 splits preserve chronological ordering!")
    print(
        "Average IS boundary gaps:",
        temporal_results["is_boundary_gaps"].mean()
    )


    # VALIDATION 3B: Autocorrelation preservation
    print("\n--- Autocorrelation Preservation ---")

    autocorr_results = validate_autocorrelation_preservation(
        T=1000,
        S=8,
        phi=0.7,
        seed=42
    )

    print(autocorr_results.to_string(index=False))


    # VALIDATION 4: PBO estimator accuracy
    print("\n--- PBO Estimator Accuracy ---")

    # Illustrative values, not empirical estimates
    estimated_pbo = [0.42, 0.68, 0.15, 0.35]
    reference_pbo = [0.45, 0.65, 0.20, 0.40]

    accuracy_results = validate_pbo_accuracy(
        estimated_pbo,
        reference_pbo
    )

    print(accuracy_results)


    # VALIDATION 5: Model-free and nonparametric sampling
    print("\n--- Model-Free / Nonparametric Validation ---")

    model_free_results = validate_model_free(
        T=1000,
        N=100,
        S=8,
        seed=42
    )

    print(model_free_results.to_string(index=False))

    # Verify every distribution was processed successfully
    assert model_free_results["no_overlap"].all()
    assert model_free_results["complete_split"].all()

    # Verify all cases generated 70 symmetric combinations
    assert (model_free_results["n_splits"] == 70).all()

    print("\nAll distribution tests passed!")








