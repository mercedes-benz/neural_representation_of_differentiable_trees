# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import pandas as pd
import py
import torch

from src.abstract.torch_model import TorchModel


@dataclass
class TorchModelMock(TorchModel):
    def get_model(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )


def test_save_load(tmpdir: py.path.local) -> None:
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.DataFrame({"abs_diff": [3, 3, 3]})
    X_validation = pd.DataFrame({"feature1": [7, 8, 9], "feature2": [10, 11, 12]})

    original_model = TorchModelMock("mock", num_targets=1, num_inputs=2, epochs=1)

    original_model.fit(X_train, y_train)
    original_model.save(tmpdir.join("test").strpath)

    loaded_model = TorchModelMock.load("mock", tmpdir.join("test").strpath)

    orig_model_predictions = original_model.predict(X_validation)
    loaded_model_predictions = loaded_model.predict(X_validation)

    assert loaded_model_predictions.equals(orig_model_predictions)
