from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense


class ConvModel:
    """
    Creates a convolutional model.
    """

    def __init__(self, kernel_size, stride, channels, pool_size, dense_units, num_classes, input_shape):
        """
        Returns a ConvModel.
        :param kernel_size: The kernel size.
        :param stride: The stride.
        :param channels: The channels.
        :param pool_size: The pool size.
        :param dense_units: The dense units.
        :param num_classes: The number of classes.
        :param input_shape: The input shape.
        """
        self.model = Sequential()

        self.model.add(
            Conv2D(channels, kernel_size=kernel_size, activation='relu', strides=stride, padding='same',
                   input_shape=input_shape)
        )
        self.model.add(
            AveragePooling2D(pool_size=pool_size)
        )
        self.model.add(Flatten())

        for dense_unit in dense_units:
            self.model.add(
                Dense(dense_unit, activation='relu')
            )
        self.model.add(Dense(num_classes, activation='softmax'))
