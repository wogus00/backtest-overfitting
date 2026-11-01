"""Read and validate the four settings for one paired experiment."""
from pathlib import Path
import re

import yaml


def load_config(filename):
    with Path(filename).open() as file:
        config = yaml.safe_load(file)
    keys = {"experiment", "seed", "blocks", "n_days"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("Config must contain exactly: experiment, seed, blocks, n_days.")
    if not isinstance(config["experiment"], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", config["experiment"]):
        raise ValueError("experiment must be a name using letters, digits, underscores or hyphens.")
    for key in ("seed", "blocks", "n_days"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer.")
    blocks, days = config["blocks"], config["n_days"]
    if blocks < 2 or blocks % 2:
        raise ValueError("blocks must be an even integer of at least 2.")
    if days < blocks or days % blocks:
        raise ValueError("n_days must be at least blocks and exactly divisible by blocks.")
    return config
