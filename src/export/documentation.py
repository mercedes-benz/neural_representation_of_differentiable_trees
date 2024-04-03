# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from src.abstract.abstract_model import AbstractModel
from src.config import Config
from src.export.logger import Logger


@dataclass
class Documentation:
    """Documentation class used for exporting model summaries"""

    config: Config
    ids: np.ndarray
    target_names: List[str]
    parent_dir: str
    timestamp: str

    def _mkdir(self, name: str) -> None:
        os.makedirs(os.path.join(self.output_folder, name), exist_ok=True)

    def __post_init__(self) -> None:
        self.output_folder = os.path.join(self.parent_dir, self.timestamp)
        self._mkdir("models")
        self._mkdir("plots")
        self._mkdir("summaries")
        self._mkdir("logs")
        self._mkdir("profiling")

    def _save_model_summary(self, model: AbstractModel) -> None:
        logger = Logger(f"{self.output_folder}/summaries/{model.name}.txt")

        logger.activate()
        model.summary()
        logger.deactivate()

    def export_models(self, models: List[AbstractModel]) -> None:
        for model in models:
            model.save(f"{self.output_folder}/models/{model.name}")
            model.plot(f"{self.output_folder}/plots/{model.name}_arch.png")
            self._save_model_summary(model)
