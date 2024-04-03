# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from math import floor
from typing import List

import pandas as pd
import tensorflow as tf

from src.models.nerdt import Nerdt, NerdtPruned
from src.models.nerdt_lib.converter import NerdtConverter


class Pruner:
    def __init__(self, nerdt: Nerdt, train_data: pd.DataFrame) -> None:
        self.nerdt = nerdt
        self.converter = NerdtConverter()
        self.tree = self.converter.to_tree(nerdt)
        self.x = tf.transpose(tf.convert_to_tensor(train_data))
        self.prev = tf.ones(shape=(1,))

    @property
    def leaf_probs(self) -> List[float]:
        return self.tree.get_leaf_probs(self.x, self.prev)

    def pruned(self, lmbda: float) -> NerdtPruned:
        probs_dict = dict(reversed(list(enumerate(self.leaf_probs))))
        sorted_probs_dict = dict(sorted(probs_dict.items(), key=lambda item: item[1]))
        indices = list(sorted_probs_dict.keys())
        cutoff = floor(len(indices) * lmbda)
        pruned = self.tree.copy().pruned(0, indices[:cutoff])

        return self.converter.to_mlp(
            pruned.expanded(),
            f"{self.nerdt.name}_{lmbda}",
            self.nerdt.timestamp,
        )
