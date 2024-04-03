# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import sys
from datetime import datetime
from typing import Any, Callable

from ray import tune

from src.abstract.abstract_model import AbstractModel
from src.config import Config
from src.export.documentation import Documentation
from src.models.nerdt import NerdtForest
from src.param_tuner import ParamTuner
from src.processing import get_data_loader
from src.validation.cv_evaluator import CvEvaluator

EPOCHS = 1_000_000_000
LR_RANGE = [0.0001, 0.001, 0.01]
DEPTH_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
PARAM_DICT = {
    "learning_rate": tune.grid_search(LR_RANGE),
    "depth": tune.grid_search(DEPTH_RANGE),
}


def get_model_factory(
    num_targets: int,
    timestamp: str,
) -> Callable[[dict[str, Any]], AbstractModel]:
    return lambda params: NerdtForest(
        name="nerdt_forest",
        num_targets=num_targets,
        epochs=EPOCHS,
        timestamp=timestamp,
        num_trees=10,
        **params,
    )


# To evaluate the NeRDT forest accuracy on the MPG data:
# python forest_tuning.py mpg


def main() -> None:
    # Parse CLI arguments
    dataset = sys.argv[1]

    # Load configuration
    config = Config(
        data_dir="./data",
        sqlite_path="./out/forest_tuning.db",
        data_set=dataset,
    )

    # Load the data
    data_loader = get_data_loader(config)
    X, y, targets, ids = data_loader.load()

    # Setup documentation object
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    documentation = Documentation(config, ids, targets, "./out", timestamp)

    # Get model constructor
    model_factory = get_model_factory(len(targets), timestamp)

    # Tune model
    tuner = ParamTuner(PARAM_DICT)
    evaluator = CvEvaluator(X, y, config, documentation)
    result = tuner.get_best_result(model_factory, evaluator)
    print(result.metrics)
    print(result.config)


if __name__ == "__main__":
    main()
