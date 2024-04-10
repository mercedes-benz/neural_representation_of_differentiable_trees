# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

from src.abstract.sklearn_model import SklearnModel


class Tree(SklearnModel):
    def get_model(self) -> BaseEstimator:
        return DecisionTreeRegressor(**self.kwargs)

    def summary(self) -> None:
        pass
