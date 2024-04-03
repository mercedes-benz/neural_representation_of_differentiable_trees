# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict

import numpy as np
import tensorflow as tf
import torch
from ray import tune
from ray.air import RunConfig
from ray.air.result import Result
from ray.tune.schedulers import AsyncHyperBandScheduler, ResourceChangingScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from src.abstract.abstract_evaluator import AbstractEvaluator
from src.abstract.abstract_model import AbstractModel


@dataclass
class ParamTuner:
    search_space: Dict[str, Any]
    max_concurrent: int = 0

    def _set_seeds(self, value: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(value)
        random.seed(value)
        np.random.seed(value)
        tf.random.set_seed(value)
        torch.manual_seed(value)

    def _objective(
        self,
        config: Dict[str, Any],
        model_constructor: Callable[[Dict[str, Any]], AbstractModel],
        evaluator: AbstractEvaluator,
    ) -> Dict[str, Any]:
        self._set_seeds(42)
        model = model_constructor(config)
        model_eval = evaluator.get_model_eval(model)
        avg_scores = model_eval.get_avg_validation_scores()
        return asdict(avg_scores)

    def get_best_result(
        self,
        model_constructor: Callable[[Dict[str, Any]], AbstractModel],
        evaluator: AbstractEvaluator,
    ) -> Result:
        tuner = tune.Tuner(
            tune.with_parameters(
                self._objective,
                model_constructor=model_constructor,
                evaluator=evaluator,
            ),
            tune_config=tune.TuneConfig(
                scheduler=ResourceChangingScheduler(
                    base_scheduler=AsyncHyperBandScheduler(
                        metric="mse",
                        mode="min",
                    ),
                ),
                search_alg=BasicVariantGenerator(
                    constant_grid_search=True,
                    max_concurrent=self.max_concurrent,
                ),
            ),
            run_config=RunConfig(
                progress_reporter=tune.CLIReporter(
                    metric="mse",
                    mode="min",
                    metric_columns=["mse", "r2"],
                    sort_by_metric=True,
                    max_progress_rows=32,
                )
            ),
            param_space=self.search_space,
        )
        return tuner.fit().get_best_result(metric="mse", mode="min")
