# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import math
import timeit
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.abstract.abstract_model import AbstractModel
from src.config import Config
from src.export.sqlite import SQLiteDB


@dataclass
class ModelTimer:
    X: pd.DataFrame
    y: pd.DataFrame
    num_samples: int
    sqlite_path: str
    timestamp: str
    config: Config
    repetitions: int = 30
    warmup: int = 10

    def __post_init__(self) -> None:
        repeats = math.ceil(self.num_samples / self.X.shape[0])
        self.X = pd.concat(
            [self.X] * repeats,
            ignore_index=True,
        ).head(self.num_samples)
        self.y = pd.concat(
            [self.y] * repeats,
            ignore_index=True,
        ).head(self.num_samples)

    def _export_time(self, model: AbstractModel, durations: List[float]) -> None:
        db = SQLiteDB(self.sqlite_path)
        db.insert_timings(
            model.info,
            self.timestamp,
            self.num_samples,
            self.repetitions,
            durations,
            self.config,
        )
        db.close()

    def time_model(self, model: AbstractModel) -> None:
        model.fit(self.X.head(10), self.y.head(10))
        _warmup = timeit.timeit(
            lambda: model.predict(self.X),
            number=self.warmup,
        )
        durations = [
            timeit.timeit(lambda: model.predict(self.X), number=1)
            for _ in range(self.repetitions)
        ]
        self._export_time(model, durations)
