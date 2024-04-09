# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from src.abstract.abstract_model import AbstractModel
from src.utils.torch_data import TorchData


class LightningWrapper(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("train_mse", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("val_mse", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@dataclass
class TorchModel(AbstractModel):
    """A concrete model class that acts as a
    wrapper for PyTorch models"""

    num_targets: int
    num_inputs: int
    epochs: int
    patience: Optional[int] = 20
    learning_rate: float = 0.001
    validation_ratio: float = 0.1
    show_progress: bool = False

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        pass

    def get_wrapper(self) -> LightningWrapper:
        return LightningWrapper(self.get_model(), self.learning_rate)

    def _get_callbacks(self) -> List[pl.Callback]:
        if self.patience is None:
            return []

        return [EarlyStopping(monitor="val_mse", patience=self.patience)]

    def __post_init__(self) -> None:
        self.model = self.get_model()
        self.wrapper = self.get_wrapper()
        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=self._get_callbacks(),
            enable_progress_bar=self.show_progress,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.x_sample = torch.tensor(
            X.head(1).values,
            dtype=torch.float32,
        )
        validation_length = int(X.shape[0] * self.validation_ratio)
        train_loader = DataLoader(
            TorchData(X.head(-validation_length), y.head(-validation_length)),
            batch_size=32,
        )
        val_loader = DataLoader(
            TorchData(X.tail(validation_length), y.tail(validation_length)),
            batch_size=32,
        )
        self.trainer.fit(
            self.wrapper,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        data_loader = DataLoader(
            TorchData(X, None),
            batch_size=32,
        )
        predictions = self.trainer.predict(self.wrapper, data_loader)
        return pd.DataFrame(torch.cat(predictions))

    def save(self, filename: str) -> None:
        torch.save(self.model.state_dict(), f"{filename}.pt")

    @classmethod
    def load(cls: Type["TorchModel"], name: str, path: str) -> "TorchModel":
        output = cls(name, num_targets=0, epochs=0, num_inputs=0, learning_rate=0)
        output.model.load_state_dict(torch.load(f"{path}.pt"), strict=False)
        output.wrapper = LightningWrapper(output.model, 0)
        return output

    def summary(self) -> None:
        pass
