# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.abstract.abstract_model import AbstractModel
from src.config import Config
from src.export.documentation import Documentation
from src.export.logger import Logger
from src.export.sqlite import SQLiteDB
from src.utils.data_types import Fold, FoldEval, ModelEval


@dataclass
class AbstractEvaluator(ABC):
    """Abstract class defining the interface of
    an evaluator used to evaluate models"""

    X: pd.DataFrame
    y: pd.DataFrame
    config: Config
    documentation: Documentation

    @property
    def ids(self) -> np.ndarray:
        return self.X.ID.unique()

    def _adjust_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_folds, y_folds = [], []

        for id in self.ids:
            X_fold = self.X[id == self.X.ID]
            y_fold = self.y[id == self.y.ID]
            min_length = min(X_fold.shape[0], y_fold.shape[0])
            X_folds.append(X_fold.head(min_length))
            y_folds.append(y_fold.head(min_length))

        return pd.concat(X_folds), pd.concat(y_folds)

    @abstractmethod
    def _get_folds(self) -> List[Fold]:
        pass

    def __post_init__(self) -> None:
        self.X, self.y = self._adjust_data()
        self.folds = self._get_folds()

    def _get_train_preds(self, model: AbstractModel, fold: Fold) -> pd.DataFrame:
        train_predictions = model.predict(fold.X_train)
        train_predictions.columns = fold.y_train.columns
        return train_predictions

    def _get_validation_preds(self, model: AbstractModel, fold: Fold) -> pd.DataFrame:
        validation_predictions = model.predict(fold.X_validation)
        validation_predictions.columns = fold.y_validation.columns
        return validation_predictions

    def _get_predictions(
        self, model: AbstractModel, fold: Fold
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_predictions = self._get_train_preds(model, fold)
        validation_predictions = self._get_validation_preds(model, fold)
        return train_predictions, validation_predictions

    def _get_fold_eval(self, model: AbstractModel, fold: Fold) -> FoldEval:
        model.fit(fold.X_train, fold.y_train)
        train_predictions, validation_predictions = self._get_predictions(model, fold)

        train_scores = model.score(fold.y_train, train_predictions)
        validation_scores = model.score(fold.y_validation, validation_predictions)
        print(f"validation mse: {validation_scores.mse}")

        return FoldEval(
            fold,
            train_predictions,
            validation_predictions,
            train_scores,
            validation_scores,
        )

    def get_model_eval(self, model: AbstractModel) -> ModelEval:
        fold_evals = [self._get_fold_eval(model, fold) for fold in self.folds]

        return ModelEval(
            model.info,
            self.config.data_set,
            fold_evals,
            self.documentation.timestamp,
        )

    def _write_eval_to_sqlite(self, model_eval: ModelEval) -> None:
        db = SQLiteDB(self.config.sqlite_path)
        db.insert_model_eval(model_eval, self.config)
        db.close()

    def _evaluate_model(self, model: AbstractModel) -> None:
        output_path = f"{self.documentation.output_folder}/logs/{model.name}.txt"
        logger = Logger(output_path)

        logger.activate()
        model_eval = self.get_model_eval(model)
        logger.deactivate()

        self._write_eval_to_sqlite(model_eval)

    def evaluate_models(self, models: List[AbstractModel]) -> None:
        [self._evaluate_model(model) for model in models]
