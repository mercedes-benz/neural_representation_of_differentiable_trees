# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from abc import abstractmethod
from typing import Any, Dict, Type

import pandas as pd
from sklearn.base import BaseEstimator

from src.abstract.abstract_model import AbstractModel


class SklearnModel(AbstractModel):
    """A concrete model class that acts as a
    wrapper for sklearn models"""

    def __init__(self, name: str, **kwargs: Dict[str, Any]) -> None:
        super().__init__(name)
        self.kwargs = kwargs
        self.model = self.get_model()

    @property
    def info(self) -> dict:
        return {**super().info, **self.kwargs}

    @abstractmethod
    def get_model(self) -> BaseEstimator:
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.input_shape = X.shape[1]
        return self.model.fit(X, self.ravel(y))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(X))

    def save(self, filename: str) -> None:
        pass
