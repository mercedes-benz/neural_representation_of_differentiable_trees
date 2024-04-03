# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str
    sqlite_path: str
    data_set: str
    evaluation_comment: str = ""
