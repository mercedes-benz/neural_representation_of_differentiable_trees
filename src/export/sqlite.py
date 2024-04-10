# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import json
import sqlite3
from dataclasses import asdict, dataclass
from os import path
from pathlib import Path
import statistics
from typing import Any, Dict, List

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
            r2_avg REAL NOT NULL,
            mae_avg REAL NOT NULL,
            mape_avg REAL NOT NULL,
            mse_avg REAL NOT NULL
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
        CREATE TABLE IF NOT EXISTS timing_evals (
            timestamp TEXT NOT NULL,
            comment TEXT NOT NULL,
            model_info TEXT NOT NULL,
            processing_info TEXT NOT NULL,
            data_set TEXT NOT NULL,
            samples REAL NOT NULL,
            repetitions REAL NOT NULL,
            duration_avg REAL NOT NULL,
            duration_std REAL NOT NULL
        );"""
        self.cursor.execute(query3)
        query4 = """
        CREATE TABLE IF NOT EXISTS timing_runs (
            timestamp TEXT NOT NULL,
            run_no REAL NOT NULL,
            duration REAL NOT NULL
        );"""
        self.cursor.execute(query4)
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
        INSERT INTO model_evals (timestamp, comment, model_info, processing_info, data_set, r2_avg, mae_avg, mape_avg, mse_avg) VALUES (
            "{model_eval.timestamp}",
            "{config.evaluation_comment}",
            "{model_info}",
            "{processing_info}",
            "{config.data_set}",
            {averaged_validation_scores.r2},
            {averaged_validation_scores.mae},
            {averaged_validation_scores.mape},
            {averaged_validation_scores.mse}
        );"""
        self.cursor.execute(query)

        for fold_eval in model_eval.fold_evals:
            self._insert_fold_eval(fold_eval, model_eval.timestamp)

        self.connection.commit()

    def insert_run(self, timestamp: str, run_no: int, duration: float) -> None:
        query = f"""
        INSERT INTO timing_runs (timestamp, run_no, duration) VALUES (
            "{timestamp}",
            {run_no},
            {duration}
        );"""
        self.cursor.execute(query)

    def insert_timings(
        self,
        model_info: Dict[str, Any],
        timestamp: str,
        num_samples: int,
        repetitions: int,
        durations: List[float],
        config: Config,
    ) -> None:
        model_info = json.dumps(model_info).replace('"', "")
        processing_info = self._get_processing_info(config)
        duration_avg = statistics.mean(durations)
        duration_std = statistics.stdev(durations)

        query = f"""
                INSERT INTO timing_evals (timestamp, comment, model_info, processing_info, data_set, samples, repetitions, duration_avg, duration_std) VALUES (
                    "{timestamp}",
                    "{config.evaluation_comment}",
                    "{model_info}",
                    "{processing_info}",
                    "{config.data_set}",
                    "{num_samples}",
                    "{repetitions}",
                    "{duration_avg}",
                    "{duration_std}"
                );"""
        self.cursor.execute(query)

        for i, duration in enumerate(durations):
            self.insert_run(timestamp, i, duration)

        self.connection.commit()

    def close(self) -> None:
        self.cursor.close()
        self.connection.close()
