# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass


@dataclass
class AccuracyScores:
    r2: float
    mae: float
    mape: float
    mse: float
