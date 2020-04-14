from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense


class ConvModel:
    """
    Creates a convolutional model.
    """

    def __init__(self, kernel_sizes, stride, channels, pool_sizes, dense_units, num_classes, input_shape):
        """
        Returns a ConvModel.
        :param kernel_sizes: The kernel sizes.
        :param stride: The stride.
        :param channels: The channels.
        :param pool_sizes: The pool sizes.
        :param dense_units: The dense units.
        :param num_classes: The number of classes.
        :param input_shape: The input shape.
        """
        self.model = Sequential()

        self.model.add(
            Conv2D(channels[0], kernel_size=kernel_sizes[0], activation='relu', strides=stride, padding='same',
                   input_shape=input_shape)
        )
        self.model.add(
            AveragePooling2D(pool_size=pool_sizes[0])
        )

        self.model.add(
            Conv2D(channels[1], kernel_size=kernel_sizes[1], activation='relu', strides=stride, padding='same',
                   input_shape=input_shape)
        )
        self.model.add(
            AveragePooling2D(pool_size=pool_sizes[1])
        )

        self.model.add(Flatten())

        for dense_unit in dense_units:
            self.model.add(
                Dense(dense_unit, activation='relu')
            )
        self.model.add(Dense(num_classes, activation='softmax'))
