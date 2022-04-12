import numpy as np
import tensorflow as tf

from data import Dataset
from model_wrappers import TensorFlowWrapper
from model_wrappers.preprocessing import BasicSequence

class NumpyTensorFlowWrapper(TensorFlowWrapper):
    def __init__(self):
        super().__init__()
        self._set_preprocessor(BasicSequence())

    def set_data(self, raw_dataset: Dataset) -> None:
        self._set_training_data(*raw_dataset.get_training_data())
        val_data = raw_dataset.get_validation_data()
        if val_data is not None:
            self._set_validation_data(*val_data)
        self._set_test_data(*raw_dataset.get_test_data())

    def _convert_to_tensorflow_dataset(self, raw_text_arr: np.ndarray, raw_label_arr: np.ndarray) -> tf.data.Dataset:
        preprocessor = self._get_preprocessor()
        text_arr = preprocessor(raw_text_arr)
        text_tensor = tf.ragged.constant(text_arr)
        label_tensor = tf.one_hot(raw_label_arr, 2)
        ds = tf.data.Dataset.from_tensor_slices((text_tensor, label_tensor))
        ds = ds.map(lambda t, l: (tf.convert_to_tensor(t), l))
        ds = ds.filter(lambda t, _: tf.shape(t)[-1] > 0)
        return ds
