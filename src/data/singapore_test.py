import os
from typing import Tuple

import numpy as np
import pandas as pd

from data import KaggleBuptGamma

DIR = os.path.dirname(__file__)

class SingaporeTest(KaggleBuptGamma):
    def __init__(self) -> None:
        super().__init__()
        fake = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'POFMAed.csv'), encoding='ISO-8859-1')
        fake = fake.rename(columns={'is_fake': 'label'})
        true = pd.read_csv(os.path.join(DIR, '..', '..', 'Data', 'sgdata.csv'), encoding='ISO-8859-1')
        true = true.rename(columns={'fake': 'label'})
        true = true.drop(columns=['ï»¿title'])
        test_df = pd.concat([fake, true])
        self._set_test_dataframe(test_df)

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        test_df = self._get_test_dataframe()
        text_arr, label_arr = self._get_data_arrays(test_df, ['text', 'label'])
        text_arr = text_arr.astype(np.str)
        return text_arr, label_arr
