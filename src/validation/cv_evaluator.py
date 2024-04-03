# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import math
from dataclasses import dataclass
from typing import List

import numpy as np

from src.abstract.abstract_evaluator import AbstractEvaluator
from src.utils.data_types import Fold


@dataclass
class CvEvaluator(AbstractEvaluator):
    """Evaluator that implements k-fold Cross Validation

    Uses 5-fold CV by default"""

    n: int = 5

    def _get_id_groups(self, ids: np.ndarray) -> List[List[str]]:
        adjusted_length = math.floor(len(ids) / self.n) * self.n
        return np.split(ids[:adjusted_length], self.n)

    def _get_folds(self) -> List[Fold]:
        if self.n == -1:
            return [Fold(self.X, self.y, i, [id]) for i, id in enumerate(self.ids)]

        id_groups = self._get_id_groups(self.ids)
        return [Fold(self.X, self.y, i, group) for i, group in enumerate(id_groups)]
