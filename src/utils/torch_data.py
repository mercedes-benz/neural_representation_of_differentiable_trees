# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TorchData(Dataset):
    def __init__(self, X: pd.DataFrame, y: Optional[pd.DataFrame]) -> None:
        self.X = torch.tensor(X.to_numpy(dtype=np.float32))
        self.y = None if y is None else torch.tensor(y.to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.y is None:
            return self.X[idx]

        return self.X[idx], self.y[idx]
