# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import torch

from DDT.interpretable_ddts.agents.ddt import DDT
from src.abstract.torch_model import TorchModel


@dataclass
class Ddt(TorchModel):
    depth: int = 3
    use_gpu: bool = True

    def get_model(self) -> torch.nn.Module:
        return DDT(
            self.num_inputs,
            None,
            None,
            2 ** (self.depth - 1),
            output_dim=self.num_targets,
            is_value=True,
            use_gpu=self.use_gpu,
        )
