from typing import Union

import numpy as np

from data import Dataset

class ModelWrapper:
    def __init__(self):
        pass

    def set_save_filepath(self, save_filepath: str) -> None:
        self._save_filepath = save_filepath

    def get_save_filepath(self) -> Union[str, None]:
        try:
            return self._save_filepath
        except AttributeError:
            return None

    def set_load_filepath(self, load_filepath: str) -> None:
        self._load_filepath = load_filepath

    def get_load_filepath(self) -> Union[str, None]:
        try:
            return self._load_filepath
        except AttributeError:
            return None

    def set_data(self, raw_dataset: Dataset) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def display_metrics(self) -> None:
        pass
