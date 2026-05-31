"""AG News RNN models: one-hot input vs Embedding input, configurable size."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from data import NUM_CLASSES, SEQ_LEN, VOCAB_SIZE


def build_onehot_rnn(rnn_units: int = 64,
                     learning_rate: float = 1e-3,
                     bidirectional: bool = False) -> keras.Model:
    """Input ids -> one-hot(VOCAB_SIZE) -> SimpleRNN -> softmax.

    One-hot blows up the input width to vocab_size on every timestep, which is
    why it is rarely used in practice. Instructive to compare against an
    Embedding-backed model with the same RNN body.
    """
    inputs = keras.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Lambda(
        lambda t: keras.ops.one_hot(t, VOCAB_SIZE),
        output_shape=(SEQ_LEN, VOCAB_SIZE),
    )(inputs)
    rnn = layers.SimpleRNN(rnn_units)
    if bidirectional:
        rnn = layers.Bidirectional(rnn)
    x = rnn(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="onehot_rnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_embedding_rnn(embed_dim: int = 64,
                        rnn_units: int = 64,
                        learning_rate: float = 1e-3,
                        bidirectional: bool = False) -> keras.Model:
    """Input ids -> Embedding -> SimpleRNN -> softmax."""
    inputs = keras.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=embed_dim,
                         mask_zero=True)(inputs)
    rnn = layers.SimpleRNN(rnn_units)
    if bidirectional:
        rnn = layers.Bidirectional(rnn)
    x = rnn(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    name = f"embedding_rnn_e{embed_dim}_u{rnn_units}"
    if bidirectional:
        name += "_bi"
    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
