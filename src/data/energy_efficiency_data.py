# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.abstract.abstract_data_loader import AbstractDataLoader
from src.utils.scaling import scale


@dataclass
class EnergyEfficiencyDataLoader(AbstractDataLoader):
    """Concrete implementation of a data loader
    for the Energy Efficiency data"""

    def __post_init__(self) -> None:
        xlsx_path = os.path.join(self.config.data_dir, "ENB2012_data.xlsx")
        self.df = pd.read_excel(xlsx_path)
        ids = list(range(self.df.shape[0]))
        self.df["ID"] = pd.Series(ids)

    @property
    def x(self) -> pd.DataFrame:
        features = self.df.drop(columns=["Y1", "Y2"])
        features["X6"] = features["X6"].astype(str)
        features["X8"] = features["X8"].astype(str)
        features = pd.get_dummies(features)
        return scale(features, "minmax")

    @property
    def y(self) -> pd.DataFrame:
        return self.df[["ID", "Y1"]]

    def preprocess_data(
        self, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return x, y
