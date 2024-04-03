# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Callable, List

import pandas as pd
import tensorflow as tf

from src.abstract.tf_model import TfModel
from src.models.nerdt_lib.layers import (
    LeafLayer,
    NerdtForestLayer,
    NerdtLayer,
    PrunedNodeLayer,
)


@dataclass
class Nerdt(TfModel):
    depth: int = 3
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.math.sigmoid

    @property
    def info(self) -> dict:
        info_dict = super().info
        info_dict.pop("activation")
        return info_dict

    def get_model(self) -> tf.keras.Model:
        self.nerdt_layer = NerdtLayer(self.depth, self.activation)
        return self.nerdt_layer


@dataclass
class NerdtForest(TfModel):
    depth: int = 3
    num_trees: int = 1
    activation: Callable[[tf.Tensor], tf.Tensor] = tf.math.sigmoid

    @property
    def info(self) -> dict:
        info_dict = super().info
        info_dict.pop("activation")
        return info_dict

    def get_model(self) -> tf.keras.Model:
        self.nerdt_layer = NerdtForestLayer(self.depth, self.num_trees, self.activation)
        return self.nerdt_layer


class NerdtPruned(TfModel):
    def __init__(
        self,
        name: str,
        timestamp: str,
        node_layers: List[PrunedNodeLayer],
        leaf_layer: LeafLayer,
        weights: List[tf.Tensor],
        biases: List[tf.Tensor],
    ) -> None:
        self.node_layers = node_layers
        self.leaf_layer = leaf_layer
        self.all_layers = node_layers + [leaf_layer]
        self.weights = weights
        self.biases = biases
        self.epochs = 1
        super().__init__(name, 1, 1, timestamp)

    @property
    def info(self) -> dict:
        info = super().info
        info["depth"] = len(self.all_layers)
        info["leaves"] = self.leaf_layer.dense.units
        info["nodes"] = sum([layer.dense.units for layer in self.all_layers])
        return info

    def get_model(self) -> tf.keras.Model:
        activation = (
            self.node_layers[0].activation if self.node_layers else tf.math.sigmoid
        )
        self.nerdt_layer = NerdtLayer(
            len(self.node_layers) + 1,
            activation,
            self.node_layers,
            self.leaf_layer,
        )
        return self.nerdt_layer

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        super().fit(X.head(10), y.head(10))
        all_layers = self.nerdt_layer.node_layers + [self.nerdt_layer.leaf_layer]

        for layer, w, b in zip(all_layers, self.weights, self.biases):
            layer.dense.set_weights((w, b))
