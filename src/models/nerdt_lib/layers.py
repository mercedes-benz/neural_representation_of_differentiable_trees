# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from typing import Any, Callable, Dict, List

import tensorflow as tf


class LeafLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_nodes: int,
    ) -> None:
        super(LeafLayer, self).__init__()
        self.prediction = tf.Variable(tf.random.normal(shape=(num_nodes,)))

    def call(self, prev: tf.Tensor) -> tf.Tensor:
        return prev * self.prediction


class NodeLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_nodes: int,
        activation: Callable[[tf.Tensor], tf.Tensor],
    ) -> None:
        super(NodeLayer, self).__init__()
        self.activation = activation
        self.dense = tf.keras.layers.Dense(num_nodes, activation=activation)

    def call(self, x: tf.Tensor, prev: tf.Tensor) -> tf.Tensor:
        p = prev * self.dense(x)
        return tf.concat([p, prev - p], axis=-1)


class PrunedNodeLayer(NodeLayer):
    def __init__(
        self,
        num_nodes: int,
        activation: Callable[[tf.Tensor], tf.Tensor],
        indices: List[int] | None = None,
    ) -> None:
        self.indices = indices
        super().__init__(num_nodes, activation)

    def call(self, x: tf.Tensor, prev: tf.Tensor) -> tf.Tensor:
        p = prev * self.dense(x)
        p_counter = tf.gather(prev - p, self.indices, axis=1)
        return tf.concat([p, p_counter], axis=-1)


class NerdtLayer(tf.keras.Model):
    def __init__(
        self,
        depth: int,
        activation: Callable[[tf.Tensor], tf.Tensor],
        node_layers: List[NodeLayer] | None = None,
        leaf_layer: LeafLayer | None = None,
    ) -> None:
        super(NerdtLayer, self).__init__()
        assert depth >= 1

        self.activation = activation
        self.depth = depth
        self.node_layers = (
            node_layers
            if node_layers
            else [NodeLayer(2**i, activation) for i in range(depth - 1)]
        )
        self.leaf_layer = leaf_layer if leaf_layer else LeafLayer(2 ** (depth - 1))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y = tf.ones(shape=(1,))

        for node_layer in self.node_layers:
            y = node_layer(x, y)

        y = self.leaf_layer(y)
        return tf.reduce_sum(y, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["depth"] = self.depth
        return config


class NerdtForestLayer(tf.keras.Model):
    def __init__(
        self,
        depth: int,
        num_trees: int,
        activation: Callable[[tf.Tensor], tf.Tensor],
        node_layers: List[NodeLayer] | None = None,
        leaf_layer: LeafLayer | None = None,
    ) -> None:
        super(NerdtForestLayer, self).__init__()
        assert depth >= 1

        self.activation = activation
        self.depth = depth
        self.num_trees = num_trees
        self.node_layers = (
            node_layers
            if node_layers
            else [NodeLayer(2**i * num_trees, activation) for i in range(depth - 1)]
        )
        self.leaf_layer = (
            leaf_layer if leaf_layer else LeafLayer(2 ** (depth - 1) * num_trees)
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y = tf.ones(shape=(1,))

        for node_layer in self.node_layers:
            y = node_layer(x, y)

        y = self.leaf_layer(x, y)
        return tf.reduce_sum(y, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["depth"] = self.depth
        config["num_trees"] = self.num_trees
        return config
