from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


class FullyConnectedModel:
    """
    Creates a Fully Connected Model.
    """

    def __init__(self, dense_units, num_classes):
        """
        Returns a FullyConnectedModel.
        :param dense_units: The dense units.
        :param num_classes: The number of classes.
        """
        self.model = Sequential()

        self.model.add(Flatten())

        # Add the fully connected layers
        for dense_unit in dense_units:
            self.model.add(
                Dense(dense_unit, activation='relu')
            )

        self.model.add(Dense(num_classes, activation='softmax'))
