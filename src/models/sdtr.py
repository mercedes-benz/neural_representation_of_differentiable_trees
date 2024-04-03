# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import torch

from SDTR.lib.sdtr import SDTR
from src.abstract.torch_model import LightningWrapper, TorchModel


class SdtrWrapper(LightningWrapper):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[0]


@dataclass
class Sdtr(TorchModel):
    num_inputs: int
    depth: int = 3
    lmbda1: float = 0.1
    lmbda2: float = 0.01

    def get_model(self) -> torch.nn.Module:
        return SDTR(
            in_features=self.num_inputs,
            num_trees=1,
            depth=self.depth - 1,
            lmbda=self.lmbda1,
            lmbda2=self.lmbda2,
            hidden_dim=None,
        )

    def get_wrapper(self) -> LightningWrapper:
        return SdtrWrapper(self.get_model(), self.learning_rate)
