# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from typing import Callable, List, Tuple

import tensorflow as tf

from src.models.nerdt import Nerdt, NerdtPruned
from src.models.nerdt_lib.layer_info import LayerInfo
from src.models.nerdt_lib.layers import LeafLayer, NerdtLayer, PrunedNodeLayer
from src.models.nerdt_lib.tree_representation import TreeRepresentation


class TreeConverter:
    def _slice_params(
        self,
        dense: tf.keras.layers.Dense,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        weights, bias = dense.get_weights()
        weight_slices: List[tf.Tensor] = tf.split(
            tf.transpose(weights),
            dense.units,
            axis=0,
        )
        bias_slices: List[tf.Tensor] = tf.split(
            tf.transpose(bias),
            dense.units,
            axis=0,
        )
        return weight_slices, bias_slices

    def _get_layer_nodes(
        self, dense: tf.keras.layers.Dense, activation: Callable[[tf.Tensor], tf.Tensor]
    ) -> List[TreeRepresentation]:
        weight_slices, bias_slices = self._slice_params(dense)
        return [
            TreeRepresentation(w, b, activation)
            for w, b in zip(weight_slices, bias_slices)
        ]

    def _get_nodes(self, nerdt_layer: NerdtLayer) -> List[List[TreeRepresentation]]:
        all_layers = nerdt_layer.node_layers + [nerdt_layer.leaf_layer]
        activation = nerdt_layer.activation
        return [self._get_layer_nodes(layer.dense, activation) for layer in all_layers]

    def _split_at_index(
        self, layer: List[TreeRepresentation], index: int
    ) -> Tuple[List[TreeRepresentation], List[TreeRepresentation]]:
        return layer[:index], layer[index:]

    def _join_layers(
        self,
        current_layer: List[TreeRepresentation],
        next_layer: List[TreeRepresentation],
    ) -> None:
        left_nodes, right_nodes = self._split_at_index(
            next_layer,
            len(current_layer),
        )

        for node, left, right in zip(current_layer, left_nodes, right_nodes):
            node.left = left
            node.right = right

    def to_tree(self, nerdt: Nerdt) -> TreeRepresentation:
        nodes = self._get_nodes(nerdt.nerdt_layer)

        for i, layer in enumerate(nodes[:-1]):
            self._join_layers(layer, nodes[i + 1])

        return nodes[0][0]


class MlpConverter:
    def _get_layer_infos(
        self, layer: List[TreeRepresentation], infos: List[LayerInfo]
    ) -> List[LayerInfo]:
        if layer == []:
            return infos

        indices = [i for i, node in enumerate(layer) if node.right]
        new_info = LayerInfo(layer, indices, layer[0].activation)

        new_lefts = [node.left for node in layer if node.left]
        new_rights = [node.right for node in layer if node.right]

        return self._get_layer_infos(new_lefts + new_rights, infos + [new_info])

    def _get_node_layers(self, infos: List[LayerInfo]) -> List[PrunedNodeLayer]:
        return [
            PrunedNodeLayer(
                num_nodes=len(info.nodes),
                activation=info.activation,
                indices=info.indices,
            )
            for info in infos
        ]

    def _get_leaf_layer(self, info: LayerInfo) -> LeafLayer:
        num_leaves = len(info.nodes)
        return LeafLayer(num_leaves)

    def _get_weights(self, infos: List[LayerInfo]) -> List[tf.Tensor]:
        return [
            tf.transpose(tf.concat([node.w for node in info.nodes], axis=0))
            for info in infos
        ]

    def _get_biases(self, infos: List[LayerInfo]) -> List[tf.Tensor]:
        return [
            tf.transpose(tf.concat([node.b for node in info.nodes], axis=0))
            for info in infos
        ]

    def to_mlp(
        self, tree: TreeRepresentation, name: str, timestamp: str
    ) -> NerdtPruned:
        infos = self._get_layer_infos([tree], [])
        node_layers = self._get_node_layers(infos[:-1])
        leaf_layer = self._get_leaf_layer(infos[-1])
        weights = self._get_weights(infos)
        biases = self._get_biases(infos)

        return NerdtPruned(
            name=name,
            timestamp=timestamp,
            node_layers=node_layers,
            leaf_layer=leaf_layer,
            weights=weights,
            biases=biases,
        )


class NerdtConverter:
    def __init__(self) -> None:
        self.mlp_converter = MlpConverter()
        self.tree_converter = TreeConverter()

    def to_tree(self, nerdt: Nerdt) -> TreeRepresentation:
        return self.tree_converter.to_tree(nerdt)

    def to_mlp(
        self, tree: TreeRepresentation, name: str, timestamp: str
    ) -> NerdtPruned:
        return self.mlp_converter.to_mlp(tree, name, timestamp)
