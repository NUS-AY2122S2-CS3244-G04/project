import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from model_wrappers.preprocessing import Isaac

class IsaacSequence(Isaac):
    def __init__(self) -> None:
        super().__init__()
        self._text_vectorization: TextVectorization = TextVectorization(standardize=None, ragged=True)

    def get_vocabulary_size(self) -> int:
        return self._text_vectorization.vocabulary_size()

    def adapt(self, raw_text_arr: np.ndarray) -> None:
        raw_text_arr = np.array(list(map(self._preprocess, raw_text_arr)), dtype='object')
        self._text_vectorization.adapt(raw_text_arr)

    def preprocess(self, raw_text_arr: np.ndarray) -> np.ndarray:
        raw_text_arr = np.array(list(map(self._preprocess, raw_text_arr)), dtype='object')
        raw_text_tensor = tf.convert_to_tensor(raw_text_arr)
        text_tensor = self._text_vectorization(raw_text_tensor)
        text_arr = text_tensor.numpy()
        return text_arr
