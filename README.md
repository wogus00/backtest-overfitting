# Backtest Overfitting

## 1. Introduction

Backtest overfitting occurs when a strategy performs well on historical
data because it fits noise rather than a persistent pattern. Testing many
strategy configurations increases the chance of selecting such a strategy.

This project explores backtest overfitting using two simulated scenarios:
a random walk without a signal and a price series with a seasonal signal.
It follows the experiments described in
[*The Probability of Backtest Overfitting*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
by Bailey et al.

## 2. Probability of Backtest Overfitting (PBO)

PBO measures how often the best-performing strategy in the training sample
ranks below the median strategy in the testing sample.

A higher PBO suggests that selecting a strategy based on its backtest
performance is less reliable. A lower PBO indicates more consistent
performance across training and testing samples, but does not guarantee
future profitability.

## 3. Combinatorially Symmetric Cross-Validation (CSCV)

CSCV evaluates strategy selection across many train/test combinations:

1. Divide the observations into an even number of blocks.
2. Use half the blocks for training and the remainder for testing.
3. Select the strategy with the highest training Sharpe ratio.
4. Evaluate its performance and rank in the testing sample.
5. Repeat for every combination.

The experiment reports PBO, performance degradation, the frequency of
negative testing Sharpe ratios, and comparisons of testing performance
distributions.

## 4. Running an Experiment

Create a Python environment and install the project and its dependencies
from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Dependencies are defined in `pyproject.toml`. The `dev` extras include Jupyter for running the walkthrough notebook.

Source code lives in the `backtest_overfitting` package under `src/`. Shared data and result locations are defined in
`src/backtest_overfitting/paths.py`.
After installation, import it in notebooks, for example:

```python
from backtest_overfitting.cscv import cscv
```

Individual steps can also be run with
`.venv/bin/python -m backtest_overfitting.main --help` and
`.venv/bin/python -m backtest_overfitting.plot --help`.

Edit `configs/config.yaml`:

```yaml
experiment: demo_experiment
seed: 42
blocks: 12
n_days: 1008
```

Then run from the repository root:

```bash
make reproduce
```

Outputs for the example above:

- `data/raw/demo_experiment/seed_42/`: original simulated price paths.
- `data/processed/demo_experiment/seed_42/`: strategy configurations and P&L matrices.
- `results/replication/demo_experiment/seed_42/tables/`: statistics and diagnostics.
- `results/replication/demo_experiment/seed_42/figures/comparison.pdf`: visual overview.
- `results/replication/demo_experiment/seed_42/experiment.json`: settings and simulation conventions.

The number of blocks must be even, and the observation count must be divisible
by it. For example, use 1,008 observations with 12 or 16 blocks.


**Results walkthrough:** See [walkthrough.ipynb](notebooks/walkthrough.ipynb)
for a brief comparison with the paper, including saved tables and plots.
To execute it locally, first run `demo_experiment` with seed `42`.