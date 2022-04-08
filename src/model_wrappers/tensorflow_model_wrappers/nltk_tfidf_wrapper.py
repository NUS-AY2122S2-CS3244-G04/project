import tensorflow as tf

from model_wrappers import CsrMatrixTensorFlowWrapper
from model_wrappers.tensorflow_model_wrappers.layers import EnhancedMLP
from model_wrappers.tensorflow_model_wrappers.metrics import F1Score

NUM_EPOCHS = 128
BATCH_SIZE = 8

NUM_HIDDEN_NODES = 8
NUM_OUTPUT_NODES = 2

class NltkTfidfWrapper(CsrMatrixTensorFlowWrapper):
    def __init__(self):
        super().__init__()

    def _create_tensorflow_model(self) -> tf.keras.Model:
        preprocessor = self._get_preprocessor()
        vocab_size = preprocessor.get_vocabulary_size()
        model = Model()
        model.build(tf.TensorShape([None, vocab_size]))
        initial_learning_rate = 1e-2
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            metrics=['accuracy', F1Score(2)]
        )
        return model

    def _get_number_of_epochs(self) -> int:
        return NUM_EPOCHS

    def _batch_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.batch(BATCH_SIZE)
        return dataset

class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = EnhancedMLP(
            NUM_OUTPUT_NODES,
            num_hidden_layers=1,
            num_hidden_nodes=NUM_HIDDEN_NODES,
            output_activation='softmax'
        )

    def call(self, x):
        x = self.mlp(x)
        return x
