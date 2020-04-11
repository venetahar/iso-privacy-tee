from keras.layers import Dense, Flatten
from keras.models import Sequential


class FullyConnectedModel:

    def __init__(self, dense_units, num_classes):
        self.model = Sequential()

        self.model.add(Flatten())

        # Add the fully connected layers
        for dense_unit in dense_units:
            self.model.add(
                Dense(dense_unit, activation='relu')
            )

        self.model.add(Dense(num_classes, activation='softmax'))
