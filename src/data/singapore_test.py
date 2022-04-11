import os
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from data import DatasetUsingDataframe

DIR = os.path.dirname(__file__)

TEST_TRAINVAL_SPLIT_FRACTION = 0.2
VAL_TRAIN_SPLIT_FRACTION = 0.2

class SingaporeTest(DatasetUsingDataframe):
    def __init__(self) -> None:
        super().__init__()
        fake = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'POFMAed.csv'))
        fake['label'] = 1
        true = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'sgdata.csv'))
        true['label'] = 0
        true = true.drop(columns=['title'])
        test_df = pd.concat([fake, true])
        self._set_test_dataframe(test_df)

    def get_training_data(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        return None

    def get_validation_data(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        return None

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        test_df = self._get_test_dataframe()
        text_arr, label_arr = self._get_data_arrays(test_df, ['text', 'label'])
        return text_arr, label_arr
