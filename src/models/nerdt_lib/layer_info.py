# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Callable, List

import tensorflow as tf

from src.models.nerdt_lib.tree_representation import TreeRepresentation


@dataclass
class LayerInfo:
    nodes: List[TreeRepresentation]
    indices: List[int]
    activation: Callable[[tf.Tensor], tf.Tensor]
