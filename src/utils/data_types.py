# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Dict, List

import pandas as pd

from src.metrics import AccuracyScores


@dataclass
class Fold:
    id: int
    validation_ids: str
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_validation: pd.DataFrame

    def __init__(
        self, X: pd.DataFrame, y: pd.DataFrame, index: int, validation_ids: List[str]
    ) -> None:
        self.id = index
        self.validation_ids = str(validation_ids)
        self.X_train = (
            X[~X.ID.isin(validation_ids)].drop(columns=["ID"]).astype("float32")
        )
        self.y_train = (
            y[~y.ID.isin(validation_ids)].drop(columns=["ID"]).astype("float32")
        )
        self.X_validation = (
            X[X.ID.isin(validation_ids)].drop(columns=["ID"]).astype("float32")
        )
        self.y_validation = (
            y[y.ID.isin(validation_ids)].drop(columns=["ID"]).astype("float32")
        )


@dataclass
class FoldEval:
    fold: Fold
    train_predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    train_scores: AccuracyScores
    validation_scores: AccuracyScores


@dataclass
class ModelEval:
    model_info: Dict[str, Any]
    data_set: str
    fold_evals: List[FoldEval]
    timestamp: str

    def get_avg_validation_scores(self) -> AccuracyScores:
        validation_scores = [
            fold_eval.validation_scores for fold_eval in self.fold_evals
        ]

        r2 = fmean([score.r2 for score in validation_scores])
        mae = fmean([score.mae for score in validation_scores])
        mape = fmean([score.mape for score in validation_scores])
        mse = fmean([score.mse for score in validation_scores])

        return AccuracyScores(r2, mae, mape, mse)
