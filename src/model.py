from tensorflow import keras

filters_size = 2 ** 6
dense_size = 2 ** 10

kernel_size_ = (3, 3)
strides_shape = (1, 1)
dense_activation = "relu"
output_activation = "softmax"
optimizer = keras.optimizers.Adam()


def get_model(num_classes: int, full_size: bool = True):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=filters_size, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.Conv2D(filters=filters_size, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Conv2D(filters=filters_size * 2, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.Conv2D(filters=filters_size * 2, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Conv2D(filters=filters_size * 4, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.Conv2D(filters=filters_size * 4, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.Conv2D(filters=filters_size * 4, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Conv2D(filters=filters_size * 8, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.Conv2D(filters=filters_size * 8, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.Conv2D(filters=filters_size * 8, kernel_size=kernel_size_, strides=strides_shape, activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(dense_size, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(dense_size, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model
