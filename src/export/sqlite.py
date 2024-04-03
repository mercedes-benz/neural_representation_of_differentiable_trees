# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import json
import sqlite3
from dataclasses import asdict, dataclass
from os import path
from pathlib import Path
from typing import Any, Dict

from src.config import Config
from src.utils.data_types import FoldEval, ModelEval


@dataclass
class SQLiteDB:
    """SQLite database connector that allows to
    save the evaluation results in SQLite format"""

    db_path: str

    def _create_tables(self) -> None:
        query1 = """
        CREATE TABLE IF NOT EXISTS model_evals (
            timestamp TEXT NOT NULL,
            comment TEXT NOT NULL,
            model_info TEXT NOT NULL,
            processing_info TEXT NOT NULL,
            data_set TEXT NOT NULL,
            r2 REAL NOT NULL,
            mae REAL NOT NULL,
            mape REAL NOT NULL,
            mse REAL NOT NULL
        );"""
        self.cursor.execute(query1)
        query2 = """
        CREATE TABLE IF NOT EXISTS fold_evals (
            timestamp TEXT NOT NULL,
            fold_id TEXT NOT NULL,
            validation_ids TEXT NOT NULL,
            train_r2 REAL NOT NULL,
            validation_r2 REAL NOT NULL,
            train_mae REAL NOT NULL,
            validation_mae REAL NOT NULL,
            train_mape REAL NOT NULL,
            validation_mape REAL NOT NULL,
            train_mse REAL NOT NULL,
            validation_mse REAL NOT NULL
        );"""
        self.cursor.execute(query2)
        query3 = """
        CREATE TABLE IF NOT EXISTS measurements (
            timestamp TEXT NOT NULL,
            model_info TEXT NOT NULL,
            samples REAL NOT NULL,
            repetitions REAL NOT NULL,
            duration REAL NOT NULL
        );"""
        self.cursor.execute(query3)
        self.connection.commit()

    def __post_init__(self) -> None:
        if not path.exists(self.db_path):
            Path(self.db_path).touch(exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self._create_tables()

    def _insert_fold_eval(self, fold_eval: FoldEval, timestamp: str) -> None:
        query = f"""
        INSERT INTO fold_evals (timestamp, fold_id, validation_ids, train_r2, validation_r2, train_mae, validation_mae, train_mape, validation_mape, train_mse, validation_mse) VALUES (
            "{timestamp}",
            "{fold_eval.fold.id}",
            "{fold_eval.fold.validation_ids}",
            {fold_eval.train_scores.r2},
            {fold_eval.validation_scores.r2},
            {fold_eval.train_scores.mae},
            {fold_eval.validation_scores.mae},
            {fold_eval.train_scores.mape},
            {fold_eval.validation_scores.mape},
            {fold_eval.train_scores.mse},
            {fold_eval.validation_scores.mse}
        );"""
        self.cursor.execute(query)

    def _get_processing_info(self, config: Config) -> str:
        config_dict = asdict(config)
        return json.dumps(config_dict).replace('"', "")

    def insert_model_eval(self, model_eval: ModelEval, config: Config) -> None:
        averaged_validation_scores = model_eval.get_avg_validation_scores()
        model_info = json.dumps(model_eval.model_info).replace('"', "")
        processing_info = self._get_processing_info(config)

        query = f"""
        INSERT INTO model_evals (timestamp, comment, model_info, processing_info, data_set, r2, mae, mape, mse) VALUES (
            "{model_eval.timestamp}",
            "{config.evaluation_comment}",
            "{model_info}",
            "{processing_info}",
            "{model_eval.data_set}",
            {averaged_validation_scores.r2},
            {averaged_validation_scores.mae},
            {averaged_validation_scores.mape},
            {averaged_validation_scores.mse}
        );"""
        self.cursor.execute(query)

        for fold_eval in model_eval.fold_evals:
            self._insert_fold_eval(fold_eval, model_eval.timestamp)

        self.connection.commit()

    def insert_measurement(
        self,
        model_info: Dict[str, Any],
        timestamp: str,
        num_samples: int,
        repetitions: int,
        duration: float,
    ) -> None:
        model_info = json.dumps(model_info).replace('"', "")
        query = f"""
                INSERT INTO measurements (timestamp, model_info, samples, repetitions, duration) VALUES (
                    "{timestamp}",
                    "{model_info}",
                    "{num_samples}",
                    "{repetitions}",
                    "{duration}"
                );"""
        self.cursor.execute(query)
        self.connection.commit()

    def close(self) -> None:
        self.cursor.close()
        self.connection.close()
