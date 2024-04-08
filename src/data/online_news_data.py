# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.abstract.abstract_data_loader import AbstractDataLoader
from src.utils.scaling import scale


@dataclass
class NewsDataLoader(AbstractDataLoader):
    """Concrete implementation of a data loader
    for the Online News Popularity data"""

    def __post_init__(self) -> None:
        data_path = os.path.join(self.config.data_dir, "OnlineNewsPopularity.csv")
        self.df = pd.read_csv(
            data_path,
            names=[
                "url",
                "timedelta",
                "n_tokens_title",
                "n_tokens_content",
                "n_unique_tokens",
                "n_non_stop_words",
                "n_non_stop_unique_tokens",
                "num_hrefs",
                "num_self_hrefs",
                "num_imgs",
                "num_videos",
                "average_token_length",
                "num_keywords",
                "data_channel_is_lifestyle",
                "data_channel_is_entertainment",
                "data_channel_is_bus",
                "data_channel_is_socmed",
                "data_channel_is_tech",
                "data_channel_is_world",
                "kw_min_min",
                "kw_max_min",
                "kw_avg_min",
                "kw_min_max",
                "kw_max_max",
                "kw_avg_max",
                "kw_min_avg",
                "kw_max_avg",
                "kw_avg_avg",
                "self_reference_min_shares",
                "self_reference_max_shares",
                "self_reference_avg_sharess",
                "weekday_is_monday",
                "weekday_is_tuesday",
                "weekday_is_wednesday",
                "weekday_is_thursday",
                "weekday_is_friday",
                "weekday_is_saturday",
                "weekday_is_sunday",
                "is_weekend",
                "LDA_00",
                "LDA_01",
                "LDA_02",
                "LDA_03",
                "LDA_04",
                "global_subjectivity",
                "global_sentiment_polarity",
                "global_rate_positive_words",
                "global_rate_negative_words",
                "rate_positive_words",
                "rate_negative_words",
                "avg_positive_polarity",
                "min_positive_polarity",
                "max_positive_polarity",
                "avg_negative_polarity",
                "min_negative_polarity",
                "max_negative_polarity",
                "title_subjectivity",
                "title_sentiment_polarity",
                "abs_title_subjectivity",
                "abs_title_sentiment_polarity",
                "shares",
            ],
        )
        ids = list(range(self.df.shape[0]))
        self.df["ID"] = pd.Series(ids)

    @property
    def x(self) -> pd.DataFrame:
        features = self.df.drop(columns=["url", "shares"])
        return scale(features, "minmax")

    @property
    def y(self) -> pd.DataFrame:
        return self.df[["ID", "shares"]]

    def preprocess_data(
        self, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return x, y
