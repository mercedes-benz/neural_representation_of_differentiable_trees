# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from typing import Optional

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def get_scaler(type: str) -> Optional[TransformerMixin]:
    if type == "standard":
        return StandardScaler()

    if type == "minmax":
        return MinMaxScaler()

    if type == "robust":
        return RobustScaler()

    return None


def scale(data: pd.DataFrame, type: str) -> pd.DataFrame:
    scaler = get_scaler(type)

    if scaler is None:
        return data

    # Don't scale ID column
    data_adjusted = data.drop(columns=["ID"])

    # Scale values
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data_adjusted),
        columns=data_adjusted.columns,
    )

    # Restore original ID column
    data_scaled["ID"] = data.ID.values

    return data_scaled
