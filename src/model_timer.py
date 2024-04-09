# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import math
import timeit
from dataclasses import dataclass

import pandas as pd

from src.abstract.abstract_model import AbstractModel
from src.export.sqlite import SQLiteDB


@dataclass
class ModelTimer:
    X: pd.DataFrame
    y: pd.DataFrame
    num_samples: int
    sqlite_path: str
    timestamp: str
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

    def _export_time(self, model: AbstractModel, run_no: int, duration: float) -> None:
        db = SQLiteDB(self.sqlite_path)
        db.insert_measurement(
            model.info,
            self.timestamp,
            self.num_samples,
            run_no,
            duration,
        )
        db.close()

    def time_model(self, model: AbstractModel) -> None:
        model.fit(self.X.head(10), self.y.head(10))
        _warmup = timeit.timeit(
            lambda: model.predict(self.X),
            number=self.warmup,
        )

        for run_number in range(self.repetitions):
            duration = timeit.timeit(lambda: model.predict(self.X), number=1)
            self._export_time(model, run_number, duration)
