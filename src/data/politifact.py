import os
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from data import DatasetUsingDataframe

DIR = os.path.dirname(__file__)

TEST_TRAINVAL_SPLIT_FRACTION = 0.2
VAL_TRAIN_SPLIT_FRACTION = 0.2

class Politifact(DatasetUsingDataframe):
    def __init__(self) -> None:
        super().__init__()
        politifact = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'PolitiFact_real_news_content.csv'))
        politifact['label'] = 0
        politifact = politifact.drop(columns=['title', 'id', 'url', 'top_img', 'authors', 'source', 'publish_date', 'movies', 'images', 'canonical_link', 'meta_data'])
        df = pd.concat([politifact])
        test_df, valtrain_df = self._split_stratified(df, TEST_TRAINVAL_SPLIT_FRACTION, 'label')
        val_df, train_df = self._split_stratified(valtrain_df, VAL_TRAIN_SPLIT_FRACTION, 'label')
        self._set_training_dataframe(train_df)
        self._set_validation_dataframe(val_df)
        self._set_test_dataframe(test_df)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        train_df = self._get_training_dataframe()
        text_arr, label_arr = self._get_data_arrays(train_df, ['text', 'label'])
        return text_arr, label_arr

    def get_validation_data(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        val_df = self._get_validation_dataframe()
        text_arr, label_arr = self._get_data_arrays(val_df, ['text', 'label'])
        return text_arr, label_arr

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        test_df = self._get_test_dataframe()
        text_arr, label_arr = self._get_data_arrays(test_df, ['text', 'label'])
        return text_arr, label_arr
