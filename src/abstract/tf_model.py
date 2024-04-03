# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Type

import pandas as pd
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from src.abstract.abstract_model import AbstractModel


@dataclass
class TfModel(AbstractModel):
    """A concrete model class that acts as a
    wrapper for TensorFlow models"""

    num_targets: int
    epochs: int
    timestamp: str
    patience: Optional[int] = 20
    verbose: int = 0
    learning_rate: float = 0.001
    validation_ratio: float = 0.1

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        pass

    def __post_init__(self) -> None:
        self.model = self.get_model()
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
            ),
        )

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        log_dir = f"./tf_logs/{self.timestamp}/{self.name}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks = [tensorboard_callback]

        if self.patience is not None:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                patience=self.patience
            )
            callbacks.append(early_stopping_callback)

        return callbacks

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        validation_length = int(X.shape[0] * self.validation_ratio)
        X_train = tf.convert_to_tensor(X.head(-validation_length))
        y_train = tf.convert_to_tensor(y.head(-validation_length))
        X_val = tf.convert_to_tensor(X.tail(validation_length))
        y_val = tf.convert_to_tensor(y.tail(validation_length))

        self.model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            callbacks=self._get_callbacks(),
            verbose=self.verbose,
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(
                tf.convert_to_tensor(X),
                verbose=self.verbose,
            )
        )

    def save(self, filename: str) -> None:
        self.model.save(f"{filename}.keras", save_format="keras")

    @classmethod
    def load(cls: Type["TfModel"], name: str, path: str) -> "TfModel":
        model = tf.keras.models.load_model(path)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = cls(name, num_targets=0, epochs=0, timestamp=timestamp)
        output.model = model
        return output

    def summary(self) -> None:
        self.model.summary()

    def plot(self, output_path: str) -> None:
        tf.keras.utils.plot_model(
            self.model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
        )

    def compute_flops(self) -> int:
        forward_pass = tf.function(
            self.model.call,
            input_signature=[tf.TensorSpec(shape=(1,) + self.model.input_shape[1:])],
        )
        graph_info = profile(
            forward_pass.get_concrete_function().graph,
            options=ProfileOptionBuilder.float_operation(),
        )
        return graph_info.total_float_ops
