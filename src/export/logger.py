# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import sys
from dataclasses import dataclass
from typing import TextIO


@dataclass
class Logger(TextIO):
    filename: str

    def __post_init__(self) -> None:
        self._original_stdout = sys.stdout
        self._file = open(self.filename, "w")  # noqa: SIM115

    def write(self, message: str) -> None:
        self._original_stdout.write(message)
        self._file.write(message)

    def flush(self) -> None:
        self._original_stdout.flush()
        self._file.flush()

    def activate(self) -> None:
        sys.stdout = self

    def deactivate(self) -> None:
        sys.stdout = self._original_stdout
