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
from src.model_timer import ModelTimer
from src.models.nerdt import Nerdt
from src.models.nerdt_lib.pruner import Pruner
from src.processing import get_data_loader
from src.validation.cv_evaluator import CvEvaluator

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


# To time a NeRDT model pruned on based on the first 4 folds:
# python pruning_timing.py mpg 5


def main() -> None:
    # Parse CLI arguments
    dataset = sys.argv[1]
    fold_num = int(sys.argv[2])

    # Load configuration
    config = Config(
        data_dir="./data",
        sqlite_path="./out/pruning_time.db",
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

    # Fit NeRDT
    evaluator = CvEvaluator(X, y, config, documentation)
    fold = evaluator.folds[fold_num]
    nerdt.fit(fold.X_train, fold.y_train)

    # Time model
    pruned = Pruner(nerdt, fold.X_train).pruned(lmbda=0.7)
    timer = ModelTimer(
        X=fold.X_validation,
        y=fold.y_validation,
        num_samples=1_000_000,
        timestamp=timestamp,
        config=config,
    )
    timer.time_model(pruned)

    documentation.export_models([nerdt, pruned])


if __name__ == "__main__":
    main()
