import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 65536
NUM_EPOCHS = 4
MAX_BATCH_SIZE = 1024
DATASET_NUM_BUCKETS = 18
DATASET_BUCKET_BOUNDARIES = [2 ** i for i in range(DATASET_NUM_BUCKETS)]
DATASET_BUCKET_BATCH_SIZES = [min(2 ** i, MAX_BATCH_SIZE) for i in range(DATASET_NUM_BUCKETS)][::-1] + [1]

EMBEDDING_DIM = 32
NUM_LSTM_UNITS = 128
NUM_HIDDEN_NODES = 512
NUM_OUTPUT_NODES = 4

def preprocess_data(text_vectorisation_layer, raw_text, raw_label):
    text = text_vectorisation_layer(raw_text)
    label = tf.one_hot(raw_label - 1, 4)
    return text, label

def load_train_data():
    train_df = pd.read_csv('./raw_data/fulltrain.csv', names=['label', 'text'])
    return train_df

def get_data_arrs(df):
    text_df = df['text']
    label_df = df['label']
    text_arr = np.array(text_df)
    label_arr = np.array(label_df)
    return text_arr, label_arr

def create_dataset(raw_text_arr, raw_label_arr, preprocess_fn, needs_shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((raw_text_arr, raw_label_arr))
    ds = ds.map(preprocess_fn)
    if needs_shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
    ds = ds.bucket_by_sequence_length(
        element_length_func=lambda text, label: tf.shape(text)[0],
        bucket_boundaries=DATASET_BUCKET_BOUNDARIES,
        bucket_batch_sizes=DATASET_BUCKET_BATCH_SIZES
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def create_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, EMBEDDING_DIM,
            # embeddings_regularizer=tf.keras.regularizers.L2()
        ),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            NUM_LSTM_UNITS,
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # recurrent_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            NUM_HIDDEN_NODES, activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            NUM_HIDDEN_NODES, activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            NUM_OUTPUT_NODES, activation='softmax',
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )
    ])
    initial_learning_rate = 1e-2
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=7e2,
        decay_rate=0.1
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['accuracy']
    )
    return model

def train(model, train_ds, save_ckpt_path=None):
    callbacks = []
    if save_ckpt_path:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_ckpt_path,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(cp_callback)
    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks
    )
    return history

def load_test_data():
    test_df = pd.read_csv('./raw_data/balancedtest.csv', names=['label', 'text'])
    return test_df

def evaluate(model, test_ds, load_ckpt_path=None):
    if load_ckpt_path:
        model.load_weights(load_ckpt_path).expect_partial()
    loss, accuracy = model.evaluate(test_ds)
    return loss, accuracy

def main(args):
    train_df = load_train_data()
    train_text_arr, train_label_arr = get_data_arrs(train_df)
    tvl = tf.keras.layers.TextVectorization()
    tvl.adapt(train_text_arr)
    preprocess_data_using_tvl = lambda raw_text, raw_label: preprocess_data(tvl, raw_text, raw_label)
    model = create_model(tvl.vocabulary_size())
    model.summary()
    if args.task in {'train', 'full'}:
        train_ds = create_dataset(train_text_arr, train_label_arr, preprocess_data_using_tvl)
        history = train(model, train_ds, save_ckpt_path=args.save_ckpt_path)
    if args.task in {'test', 'full'}:
        test_df = load_test_data()
        test_text_arr, test_label_arr = get_data_arrs(test_df)
        test_ds = create_dataset(test_text_arr, test_label_arr, preprocess_data_using_tvl)
        loss, accuracy = evaluate(model, test_ds, load_ckpt_path=args.load_ckpt_path)
        print('Loss:', loss)
        print('Accuracy:', accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='task')

    train_parser = subparser.add_parser('train')
    train_parser.add_argument('--save_ckpt_path', type=str)

    test_parser = subparser.add_parser('test')
    test_parser.add_argument('--load_ckpt_path', type=str)

    full_parser = subparser.add_parser('full')
    full_parser.add_argument('--save_ckpt_path', type=str)
    full_parser.add_argument('--load_ckpt_path', type=str)

    args = parser.parse_args()
    main(args)
