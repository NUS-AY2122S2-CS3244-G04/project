from typing import List

import tensorflow as tf

from model_wrappers import NumpyTensorFlowWrapper
from model_wrappers.preprocessing import NltkTokenStopLemmaSequence
from model_wrappers.tensorflow_model_wrappers.layers import EnhancedMLP
from model_wrappers.tensorflow_model_wrappers.layers import EnhancedStackedRNN
from model_wrappers.tensorflow_model_wrappers.metrics import F1Score

NUM_EPOCHS = 128
EARLY_STOPPING_PATIENCE = 10
INITIAL_LEARNING_RATE = 1e-2

EMBEDDING_DIM = 4
NUM_RNN_UNITS = 4
NUM_OUTPUT_NODES = 2

class NltkRnnWrapper(NumpyTensorFlowWrapper):
    def __init__(self):
        super().__init__()
        self._set_preprocessor(NltkTokenStopLemmaSequence())

    def _create_tensorflow_model(self) -> tf.keras.Model:
        preprocessor = self._get_preprocessor()
        vocab_size = preprocessor.get_vocabulary_size()
        model = Model(vocab_size)
        model.build(tf.TensorShape([None, None]))
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
            metrics=['accuracy', F1Score(2)]
        )
        return model

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = super()._get_callbacks()
        callbacks.extend([
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                patience=2,
                min_delta=1e-4,
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_macro',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_micro',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_1',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_2',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            )
        ])
        return callbacks

    def _get_number_of_epochs(self) -> int:
        return NUM_EPOCHS

class Model(tf.keras.Model):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            mask_zero=True
        )
        self.rnn = EnhancedStackedRNN(
            lambda num_nodes, *lc_args, **lc_kwargs: tf.keras.layers.LSTM(
                num_nodes,
                *lc_args,
                # kernel_regularizer=tf.keras.regularizers.L2(),
                # recurrent_regularizer=tf.keras.regularizers.L2(),
                # bias_regularizer=tf.keras.regularizers.L2(),
                **lc_kwargs
            ),
            NUM_RNN_UNITS,
            has_batch_norm=True,
            is_bidirectional=True
        )
        self.mlp = EnhancedMLP(
            NUM_OUTPUT_NODES,
            output_activation='softmax',
            # dropout_rate=0.5,
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.mlp(x)
        return x
