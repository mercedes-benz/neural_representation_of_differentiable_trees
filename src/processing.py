# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from src.abstract.abstract_data_loader import AbstractDataLoader
from src.config import Config
from src.data.abalone_data import AbaloneDataLoader
from src.data.energy_efficiency_data import EnergyEfficiencyDataLoader
from src.data.mpg_data import MpgDataLoader


def get_data_loader(config: Config) -> AbstractDataLoader:
    if config.data_set == "mpg":
        return MpgDataLoader(config)

    if config.data_set == "energy_efficiency":
        return EnergyEfficiencyDataLoader(config)

    if config.data_set == "abalone":
        return AbaloneDataLoader(config)

    raise ValueError(f"Invalid data set: {config.data_set}")
