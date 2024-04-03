# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import math
from functools import reduce
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from src.utils.data_types import ModelEval


def filename_contains_substring(filename: str, substrings: List[str]) -> bool:
    path = Path(filename)
    return reduce(
        lambda acc, substring: substring in path.stem or acc, substrings, False
    )


def filter_filenames(filenames: List[str], substrings: List[str]) -> List[str]:
    return [
        filename
        for filename in filenames
        if filename_contains_substring(filename, substrings)
    ]


def sort_df_columns_by_name(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(sorted(df.columns), axis=1)


def get_unique_values(elements: List[str]) -> List[str]:
    return sorted(list(set(elements)))


def remove_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]


def flatten_2d_list(list_of_lists: List[List[Any]]) -> List[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def combine_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    dfs_modified = [df.reset_index(drop=True) for df in dfs]
    combined_df = remove_duplicate_cols(pd.concat(dfs_modified, axis=1))
    combined_df.index = dfs[0].index
    return combined_df


def compress_second_dim(shape: List[int]) -> List[int]:
    return shape[:1] + [1] + shape[2:]


def split_df_into_chunks(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    num_chunks = math.ceil(len(df) / chunk_size)
    return [
        df.iloc[i * chunk_size : (i + 1) * chunk_size, :] for i in range(num_chunks)
    ]


def add_suffix_to_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    df.columns = [f"{column}_{suffix}" for column in df.columns]
    return df


def get_column_names(model_eval: ModelEval, fold_index: int) -> List[str]:
    fold = model_eval.fold_evals[fold_index].fold
    target_names = fold.y_train.columns
    model_name = model_eval.model_info["name"]
    return [f"target_{i}_{model_name}" for i, _ in enumerate(target_names)]


def get_model_predictions(
    model_eval: ModelEval, fold_index: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_predictions = model_eval.fold_evals[fold_index].train_predictions
    validation_predictions = model_eval.fold_evals[fold_index].validation_predictions

    column_names = get_column_names(model_eval, fold_index)
    train_predictions.columns = column_names
    validation_predictions.columns = column_names

    return train_predictions, validation_predictions


def get_fold_predictions(
    model_evals: List[ModelEval], fold_index: int
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    predictions = [
        get_model_predictions(model_eval, fold_index) for model_eval in model_evals
    ]
    train_preds = [train_pred for train_pred, _ in predictions]
    validation_preds = [validation_pred for _, validation_pred in predictions]
    return train_preds, validation_preds


def get_targets(
    model_evals: List[ModelEval], fold_index: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_train = model_evals[0].fold_evals[fold_index].fold.y_train
    y_validation = model_evals[0].fold_evals[fold_index].fold.y_validation

    column_names = [f"target_{i}" for i in range(y_validation.shape[1])]
    y_train.columns = column_names
    y_validation.columns = column_names

    return y_train, y_validation


def combine_dfs_with_separator(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    placeholder = pd.DataFrame({"": 0}, index=left.index)
    return combine_dfs([left, placeholder, right])
