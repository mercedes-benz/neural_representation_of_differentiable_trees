# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import sys
from datetime import datetime
from typing import Any, Callable

from ray import tune

from src.abstract.abstract_model import AbstractModel
from src.config import Config
from src.export.documentation import Documentation
from src.models.ddt import Ddt
from src.models.gbdt import Gbdt
from src.models.nerdt import Nerdt
from src.models.node import Node
from src.models.sdtr import Sdtr
from src.models.tel import Tel
from src.models.tree import Tree
from src.param_tuner import ParamTuner
from src.processing import get_data_loader
from src.validation.cv_evaluator import CvEvaluator

EPOCHS = 1_000_000_000
LR_RANGE = [0.0001, 0.001, 0.01]
DEPTH_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
PARAM_DICT = {
    "dt": {
        "criterion": tune.grid_search(
            ["squared_error", "friedman_mse", "absolute_error", "poisson"]
        ),
        "max_depth": tune.grid_search(DEPTH_RANGE),
    },
    "gbdt": {
        "criterion": tune.grid_search(["squared_error", "friedman_mse"]),
        "learning_rate": tune.grid_search([0.25, 0.1, 0.01]),
        "n_estimators": tune.grid_search([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        "max_depth": tune.grid_search(DEPTH_RANGE),
    },
    "nerdt": {
        "learning_rate": tune.grid_search(LR_RANGE),
        "depth": tune.grid_search(DEPTH_RANGE),
    },
    "ddt": {
        "learning_rate": tune.grid_search(LR_RANGE),
        "depth": tune.grid_search(DEPTH_RANGE),
    },
    "tel": {
        "smooth_step_param": tune.grid_search([1, 0.1, 0.01, 0.001, 0.0001]),
        "learning_rate": tune.grid_search(LR_RANGE),
        "depth": tune.grid_search(DEPTH_RANGE),
    },
    "sdtr": {
        "learning_rate": tune.grid_search(LR_RANGE),
        "depth": tune.grid_search(DEPTH_RANGE),
        "lmbda1": tune.grid_search([0.1, 0.01, 0.001]),
        "lmbda2": tune.grid_search([0.1, 0.01, 0.001]),
    },
    "node": {
        "learning_rate": tune.grid_search(LR_RANGE),
        "depth": tune.grid_search(DEPTH_RANGE),
    },
}


def get_model_factory(
    name: str,
    num_inputs: int,
    num_targets: int,
    timestamp: str,
) -> Callable[[dict[str, Any]], AbstractModel]:
    if name == "dt":
        return lambda params: Tree(
            name=name,
            **params,
        )
    if name == "gbdt":
        return lambda params: Gbdt(
            name=name,
            **params,
        )
    if name == "ddt":
        return lambda params: Ddt(
            name=name,
            num_targets=num_targets,
            epochs=EPOCHS,
            num_inputs=num_inputs,
            use_gpu=False,
            **params,
        )
    if name == "sdtr":
        return lambda params: Sdtr(
            name=name,
            num_targets=num_targets,
            epochs=EPOCHS,
            num_inputs=num_inputs,
            **params,
        )
    if name == "tel":
        return lambda params: Tel(
            name=name,
            num_targets=num_targets,
            epochs=EPOCHS,
            timestamp=timestamp,
            **params,
        )
    if name == "node":
        return lambda params: Node(
            name=name,
            num_targets=num_targets,
            epochs=EPOCHS,
            num_inputs=num_inputs,
            **params,
        )
    if name == "nerdt":
        return lambda params: Nerdt(
            name=name,
            num_targets=num_targets,
            epochs=EPOCHS,
            timestamp=timestamp,
            **params,
        )

    raise ValueError(f"Invalid model name {name}")


# Our model on the MPG data:
# python tuning.py mpg nerdt


def main() -> None:
    # Parse CLI arguments
    dataset = sys.argv[1]
    name = sys.argv[2]

    # Load configuration
    config = Config(
        data_dir="./data",
        sqlite_path="./out/tuning.db",
        data_set=dataset,
    )

    # Load the data
    data_loader = get_data_loader(config)
    X, y, targets, ids = data_loader.load()

    # Setup documentation object
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    documentation = Documentation(config, ids, targets, "./out", timestamp)

    # Get model constructor
    model_factory = get_model_factory(
        name=name,
        num_inputs=X.shape[1] - 1,
        num_targets=len(targets),
        timestamp=timestamp,
    )

    # Tune model
    tuner = ParamTuner(PARAM_DICT[name])
    evaluator = CvEvaluator(X, y, config, documentation)
    result = tuner.get_best_result(model_factory, evaluator)
    print(result.metrics)
    print(result.config)


if __name__ == "__main__":
    main()
