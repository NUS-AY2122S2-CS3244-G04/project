import os
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from data import BuptGamma

DIR = os.path.dirname(__file__)

VALIDATION_FRACTION = 0.2

class BuptGammaWithValidation(BuptGamma):
    def __init__(self) -> None:
        super().__init__()
        train_df = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'fulltrain.csv'), names=['label', 'text'])
        train_df.loc[train_df['label'] < 4, 'label'] = 1
        train_df.loc[train_df['label'] == 4, 'label'] = 0
        val_df, train_df = self._split_stratified(train_df, VALIDATION_FRACTION, 'label')
        self._set_training_dataframe(train_df)
        self._set_validation_dataframe(val_df)
        test_df = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'balancedtest.csv'), names=['label', 'text'])
        test_df.loc[test_df['label'] < 4, 'label'] = 1
        test_df.loc[test_df['label'] == 4, 'label'] = 0
        self._set_test_dataframe(test_df)

    def get_validation_data(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        val_df = self._get_validation_dataframe()
        text_arr, label_arr = self._get_data_arrays(val_df, ['text', 'label'])
        return text_arr, label_arr
