# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import tensorflow as tf
from tf_trees import TEL

from src.abstract.tf_model import TfModel


@dataclass
class Tel(TfModel):
    depth: int = 3
    smooth_step_param: float = 0.0001

    def get_model(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            [
                TEL(
                    output_logits_dim=self.num_targets,
                    depth=self.depth,
                    smooth_step_param=self.smooth_step_param,
                ),
            ]
        )

    def save(self, filename: str) -> None:
        # Export to tf format
        self.model.save(f"{filename}", save_format="tf")
