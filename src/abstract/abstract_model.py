# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.metrics import AccuracyScores


@dataclass
class AbstractModel(ABC):
    """Abstract model class that defines a common interface for
    all models which can be used for training etc."""

    name: str

    @property
    def info(self) -> dict:
        return asdict(self)

    def ravel(self, y: pd.DataFrame) -> pd.DataFrame:
        return y.values.ravel() if y.shape[1] == 1 else y

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> AccuracyScores:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = 100 * mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return AccuracyScores(r2, mae, mape, mse)

    @abstractmethod
    def save(self, filename: str) -> None:
        pass

    @abstractmethod
    def summary(self) -> None:
        pass
