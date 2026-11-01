#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec .venv/bin/python -m backtest_overfitting.run_experiments --config configs/config.yaml
