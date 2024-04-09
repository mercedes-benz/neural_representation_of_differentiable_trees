# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import torch

from NODE.lib.odst import ODST
from src.abstract.torch_model import TorchModel


@dataclass
class Node(TorchModel):
    num_inputs: int
    depth: int = 3

    def get_model(self) -> torch.nn.Module:
        return ODST(
            in_features=self.num_inputs,
            num_trees=1,
            depth=self.depth - 1,
        )
