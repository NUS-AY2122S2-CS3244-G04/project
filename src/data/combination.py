from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from data import DatasetUsingDataframe

class Combination(DatasetUsingDataframe):
    def __init__(self, datasets) -> None:
        super().__init__()
        train_df = pd.concat([dataset._get_training_dataframe() for dataset in datasets])
        val_df = pd.concat([dataset._get_validation_dataframe() for dataset in datasets])
        test_df = pd.concat([dataset._get_test_dataframe() for dataset in datasets])
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
