# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Tuple

from src.models.nerdt import Nerdt, NerdtPruned
from src.models.nerdt_lib.pruner import Pruner
from src.utils.data_types import Fold, FoldEval, ModelEval
from src.validation.cv_evaluator import CvEvaluator


@dataclass
class PruningEvaluator(CvEvaluator):
    """Special pruning evaluator"""

    def evaluted_pruned_on_fold(
        self,
        pruned: NerdtPruned,
        fold: Fold,
    ) -> FoldEval:
        pruned.fit(fold.X_train, fold.y_train)
        train_predictions, validation_predictions = self._get_predictions(pruned, fold)

        train_scores = pruned.score(fold.y_train, train_predictions)
        validation_scores = pruned.score(fold.y_validation, validation_predictions)
        print(f"validation mse: {validation_scores.mse}")

        return FoldEval(
            fold,
            train_predictions,
            validation_predictions,
            train_scores,
            validation_scores,
        )

    def evaluate_pruning_on_fold(
        self,
        model: Nerdt,
        fold: Fold,
        lmbdas: List[float],
    ) -> Dict[float, Tuple[Dict[str, Any], FoldEval]]:
        model.fit(fold.X_train, fold.y_train)

        pruner = Pruner(model, fold.X_train)
        pruned_models = [pruner.pruned(lmbda) for lmbda in lmbdas]

        pruned_infos = [pruned_model.info for pruned_model in pruned_models]
        fold_evals: List[FoldEval] = [
            self.evaluted_pruned_on_fold(pruned_model, fold)
            for pruned_model in pruned_models
        ]

        return {
            lmbda: (info, eval)
            for lmbda, info, eval in zip(lmbdas, pruned_infos, fold_evals)
        }

    def evaluate_pruning(self, model: Nerdt, lmbdas: List[float]) -> None:
        results = [
            self.evaluate_pruning_on_fold(model, fold, lmbdas) for fold in self.folds
        ]

        for lmbda in lmbdas:
            nodes = [result[lmbda][0]["nodes"] for result in results]
            leaves = [result[lmbda][0]["leaves"] for result in results]
            evals = [result[lmbda][1] for result in results]

            info = model.info
            info["nodes"] = mean(nodes)
            info["leaves"] = mean(leaves)
            info["lambda"] = lmbda

            model_eval = ModelEval(
                info, self.config.data_set, evals, self.documentation.timestamp
            )
            self._write_eval_to_sqlite(model_eval)
