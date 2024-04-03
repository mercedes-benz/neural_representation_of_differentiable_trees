# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import Config


@dataclass
class AbstractDataLoader(ABC):
    """Abstract class defining the
    interface of a data loader"""

    config: Config

    @abstractproperty
    def x(self) -> pd.DataFrame:
        pass

    @abstractproperty
    def y(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def preprocess_data(
        self, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray]:
        x, y = self.preprocess_data(self.x, self.y)

        target_names = list(y.columns.drop("ID"))
        ids = y.ID.unique()

        return x, y, target_names, ids
