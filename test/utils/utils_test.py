# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import numpy as np
import pandas as pd

from src.utils.utils import (
    filename_contains_substring,
    flatten_2d_list,
    get_unique_values,
    remove_duplicate_cols,
    sort_df_columns_by_name,
)


def test_filename_contains_substring() -> None:
    assert filename_contains_substring("abc.txt", ["abc"])
    assert filename_contains_substring("abc.txt", ["efg", "abc"])
    assert not filename_contains_substring("abc.txt", ["txt"])
    assert not filename_contains_substring("dir/abc.txt", ["dir"])


def test_sort_df_columns_by_name() -> None:
    df = pd.DataFrame({"colB": [1, 2], "colA": [3, 4]})
    reference = pd.DataFrame({"colA": [3, 4], "colB": [1, 2]})
    result = sort_df_columns_by_name(df)
    assert result.equals(reference)


def test_get_unique_values() -> None:
    assert get_unique_values([1, 2, 3, 2, 1]) == [1, 2, 3]


def test_remove_duplicate_cols() -> None:
    df = pd.DataFrame(
        np.array([[1, 1, 4, 7], [2, 2, 5, 8], [3, 3, 6, 9]]),
        columns=["col1", "col2", "col3", "col1"],
    )
    reference = pd.DataFrame(
        np.array([[1, 1, 4], [2, 2, 5], [3, 3, 6]]),
        columns=["col1", "col2", "col3"],
    )
    result = remove_duplicate_cols(df)
    assert result.equals(reference)


def test_flatten_2d_list() -> None:
    assert flatten_2d_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]
