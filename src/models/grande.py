# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import torch

from GRANDE.GRANDE import GRANDE
from src.abstract.tf_model import TfModel


@dataclass
class Node(TfModel):
    depth: int = 3

    def get_model(self) -> torch.nn.Module:
        return GRANDE(
            depth=self.depth,
            n_estimators=1,
        )
