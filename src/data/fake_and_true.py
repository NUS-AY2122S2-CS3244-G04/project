import os
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from data import DatasetUsingDataframe

DIR = os.path.dirname(__file__)

TEST_TRAINVAL_SPLIT_FRACTION = 0.2
VAL_TRAIN_SPLIT_FRACTION = 0.2

class FakeAndTrue(DatasetUsingDataframe):
    def __init__(self) -> None:
        super().__init__()
        fake = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'Fake.csv'))
        fake['text'] = fake['text'].str.strip()
        fake.loc[fake['text'] == '', 'text'] = fake['title']
        fake['label'] = 1
        fake = fake.drop(columns=['title', 'subject', 'date'])
        true = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'True.csv'))
        true['text'] = true['text'].str.strip()
        true.loc[true['text'] == '', 'text'] = true['title']
        true['label'] = 0
        true = true.drop(columns=['title', 'subject', 'date'])
        df = pd.concat([fake, true])
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
