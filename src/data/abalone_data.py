# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.abstract.abstract_data_loader import AbstractDataLoader
from src.utils.scaling import scale


@dataclass
class AbaloneDataLoader(AbstractDataLoader):
    """Concrete implementation of a data loader
    for the Abalone data"""

    def __post_init__(self) -> None:
        data_path = os.path.join(self.config.data_dir, "abalone.data")
        self.df = pd.read_csv(
            data_path,
            names=[
                "Sex",
                "Length",
                "Diameter",
                "Height",
                "Whole_Weight",
                "Shucked_Weight",
                "Viscera_Weight",
                "Shell_Weight",
                "Rings",
            ],
        )
        ids = list(range(self.df.shape[0]))
        self.df["ID"] = pd.Series(ids)

    @property
    def x(self) -> pd.DataFrame:
        features = self.df.drop(columns=["Rings"])
        features["Sex"] = features["Sex"].astype(str)
        features = pd.get_dummies(features)
        return scale(features, "minmax")

    @property
    def y(self) -> pd.DataFrame:
        return self.df[["ID", "Rings"]]

    def preprocess_data(
        self, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return x, y
