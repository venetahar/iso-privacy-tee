from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


class FullyConnectedModel:
    """
    Creates a Fully Connected Model.
    """

    def __init__(self, dense_units, num_classes, input_shape):
        """
        Returns a FullyConnectedModel.
        :param dense_units: The dense units.
        :param num_classes: The number of classes.
        :param input_shape: The input shape.
        """
        self.model = Sequential()

        self.model.add(Flatten(input_shape=input_shape))

        for dense_unit in dense_units:
            self.model.add(
                Dense(dense_unit, activation='relu')
            )

        self.model.add(Dense(num_classes, activation='softmax'))
