# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import py
import tensorflow as tf

from src.abstract.tf_model import TfModel


@dataclass
class TfModelMock(TfModel):
    def get_model(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(self.num_targets),
            ]
        )


def test_save_load(tmpdir: py.path.local) -> None:
    X_train = pd.DataFrame(
        {
            "feature1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "feature2": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        }
    )
    y_train = pd.DataFrame({"abs_diff": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]})
    X_validation = pd.DataFrame({"feature1": [10, 11, 12], "feature2": [20, 21, 22]})

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    original_model = TfModelMock(
        "mock",
        num_targets=1,
        epochs=1,
        timestamp=timestamp,
    )

    original_model.fit(X_train, y_train)
    original_model.save(tmpdir.join("test").strpath)
    loaded_model = TfModelMock.load("mock", tmpdir.join("test.keras").strpath)

    orig_model_predictions = original_model.predict(X_validation)
    loaded_model_predictions = loaded_model.predict(X_validation)

    assert loaded_model_predictions.equals(orig_model_predictions)
