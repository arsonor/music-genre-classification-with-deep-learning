import keras
import tensorflow as tf


def build_model(input_shape, l2_reg, learning_rate):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
            keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
            keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                128,
                (2, 2),
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
            keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dense(
                128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)
            ),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
