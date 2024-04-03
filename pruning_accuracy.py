# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
import random
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.config import Config
from src.export.documentation import Documentation
from src.models.nerdt import Nerdt
from src.processing import get_data_loader
from src.validation.pruning_evaluator import PruningEvaluator

PARAM_DICT = {
    "mpg": {
        "depth": 4,
        "learning_rate": 0.001,
    },
    "energy_efficiency": {
        "depth": 10,
        "learning_rate": 0.01,
    },
    "abalone": {
        "depth": 8,
        "learning_rate": 0.01,
    },
}


def set_seeds(value: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)


# To evaluate the pruning accuracy on the MPG data:
# python pruning_accuracy.py mpg


def main() -> None:
    # Parse CLI arguments
    dataset = sys.argv[1]

    # Load configuration
    config = Config(
        data_dir="./data",
        sqlite_path="./out/pruning_acc.db",
        data_set=dataset,
    )

    # Load the data
    data_loader = get_data_loader(config)
    X, y, targets, ids = data_loader.load()

    # Setup documentation object
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    documentation = Documentation(config, ids, targets, "./out", timestamp)

    # Define NeRDT
    set_seeds(42)
    best_params = PARAM_DICT[config.data_set]
    nerdt = Nerdt(
        name="nerdt",
        num_targets=len(targets),
        epochs=1_000_000_000,
        timestamp=timestamp,
        **best_params,
    )

    # Evaluate pruning
    evaluator = PruningEvaluator(X, y, config, documentation)
    evaluator.evaluate_pruning(
        nerdt, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
    )


if __name__ == "__main__":
    main()
