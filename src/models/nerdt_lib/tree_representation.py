# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import tensorflow as tf


@dataclass
class TreeRepresentation:
    w: tf.Tensor
    b: tf.Tensor
    activation: Callable[[tf.Tensor], tf.Tensor]
    left: Optional["TreeRepresentation"] = None
    right: Optional["TreeRepresentation"] = None

    @property
    def is_leaf(self) -> bool:
        return not self.left and not self.right

    @property
    def num_nodes(self) -> int:
        left_nodes = self.left.num_nodes if self.left else 0
        right_nodes = self.right.num_nodes if self.right else 0
        return left_nodes + right_nodes + 1

    @property
    def num_leaves(self) -> int:
        left_leaves = self.left.num_leaves if self.left else 0
        right_leaves = self.right.num_leaves if self.right else 0
        return max(left_leaves + right_leaves, 1)

    @property
    def depth(self) -> int:
        left_depth = self.left.depth if self.left else 0
        right_depth = self.right.depth if self.right else 0
        return max(left_depth, right_depth) + 1

    def _get_probs(self, x: tf.Tensor, prev: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        p = self.activation(self.w @ x + self.b) * prev
        return p, prev - p

    def get_leaf_probs(self, x: tf.Tensor, prev: tf.Tensor) -> List[float]:
        if self.is_leaf:
            return [float(tf.reduce_mean(prev).numpy())]

        p_l, p_r = self._get_probs(x, prev)
        left_probs = self.left.get_leaf_probs(x, p_l) if self.left else []
        right_probs = self.right.get_leaf_probs(x, p_r) if self.right else []
        return left_probs + right_probs

    def pruned(
        self, index: int, indices_to_prune: List[int]
    ) -> Optional["TreeRepresentation"]:
        if self.is_leaf:
            return None if index in indices_to_prune else self

        self.left = self.left.pruned(index * 2, indices_to_prune)
        self.right = self.right.pruned(index * 2 + 1, indices_to_prune)

        if self.left and self.right:
            return self

        if self.left and not self.right:
            return self.left

        if self.right and not self.left:
            return self.right

        return None

    def insert_node(self, child: "TreeRepresentation") -> "TreeRepresentation":
        new_w = tf.zeros(shape=self.w.shape)
        new_b = tf.cast(tf.fill(self.b.shape, 1_000), tf.float32)
        return TreeRepresentation(new_w, new_b, self.activation, left=child)

    def expanded(self) -> "TreeRepresentation":
        if self.left:
            self.left = self.left.expanded()

        if self.right:
            self.right = self.right.expanded()

        if not self.left or not self.right or self.left.depth == self.right.depth:
            return self

        if self.left.depth < self.right.depth:
            self.left = self.insert_node(self.left)

        if self.left.depth > self.right.depth:
            self.right = self.insert_node(self.right)

        return self.expanded()

    def copy(self) -> "TreeRepresentation":
        return TreeRepresentation(
            w=tf.identity(self.w),
            b=tf.identity(self.b),
            activation=self.activation,
            left=self.left.copy() if self.left else None,
            right=self.right.copy() if self.right else None,
        )
