# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import sys
from datetime import datetime

from src.abstract.abstract_model import AbstractModel
from src.config import Config
from src.export.documentation import Documentation
from src.model_timer import ModelTimer
from src.models.ddt import Ddt
from src.models.grande import Grande
from src.models.nerdt import Nerdt
from src.models.node import Node
from src.models.sdtr import Sdtr
from src.models.tel import Tel
from src.processing import get_data_loader


def get_model(
    name: str,
    depth: int,
    num_inputs: int,
    num_targets: int,
    timestamp: str,
) -> AbstractModel:
    if name == "ddt":
        return Ddt(
            name="ddt",
            num_targets=num_targets,
            epochs=1,
            num_inputs=num_inputs,
            depth=depth,
        )
    if name == "sdtr":
        return Sdtr(
            name="sdtr",
            num_targets=num_targets,
            epochs=1,
            num_inputs=num_inputs,
            depth=depth,
        )
    if name == "tel":
        return Tel(
            name="tel",
            num_targets=num_targets,
            epochs=1,
            timestamp=timestamp,
            depth=depth,
        )
    if name == "node":
        return Node(
            name="node",
            num_targets=num_targets,
            epochs=1,
            num_inputs=num_inputs,
            depth=depth,
        )
    if name == "grande":
        return Grande(
            name="grande",
            num_targets=num_targets,
            epochs=1,
            timestamp=timestamp,
            depth=depth,
        )
    if name == "nerdt":
        return Nerdt(
            name="nerdt",
            num_targets=num_targets,
            epochs=1,
            timestamp=timestamp,
            depth=depth,
        )

    raise ValueError(f"Invalid model name {name}")


# Our model with depth 3 on the MPG data:
# python timing.py mpg nerdt 3

# DDT with depth 8 on the Abalone data:
# python timing.py abalone ddt 8


def main() -> None:
    # Parse CLI arguments
    dataset = sys.argv[1]
    name = sys.argv[2]
    depth = int(sys.argv[3])

    # Load configuration
    config = Config(
        data_dir="./data",
        sqlite_path="./out/timing.db",
        data_set=dataset,
    )

    # Load the data
    data_loader = get_data_loader(config)
    X, y, targets, ids = data_loader.load()

    # Setup documentation object
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    documentation = Documentation(config, ids, targets, "./out", timestamp)

    # Define model
    model = get_model(name, depth, X.shape[1] - 1, len(targets), timestamp)

    # Time models
    timer = ModelTimer(
        X=X.drop(columns=["ID"]).astype("float32"),
        y=y.drop(columns=["ID"]).astype("float32"),
        num_samples=1_000_000,
        timestamp=timestamp,
        config=config,
    )
    timer.time_model(model)

    # Export models
    documentation.export_models([model])


if __name__ == "__main__":
    main()
