# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

from src.abstract.sklearn_model import SklearnModel


class Gbdt(SklearnModel):
    def get_model(self) -> BaseEstimator:
        return GradientBoostingRegressor(**self.kwargs)

    def summary(self) -> None:
        pass
