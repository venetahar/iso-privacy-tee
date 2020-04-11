from keras.datasets import mnist

# the data, split between train and test sets
from keras.utils import to_categorical

from common.constants import DENSE_UNITS, NUM_CLASSES, TRAINING_PARAMS, MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS
from common.model_training import ModelTraining
from mnist.models.fully_connected_model import FullyConnectedModel


def train_mnist_model():
    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
    training_data, training_labels = prepare_data(training_data, training_labels)
    test_data, test_labels = prepare_data(test_data, test_labels)

    model = FullyConnectedModel(DENSE_UNITS, NUM_CLASSES).model
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train(training_data, training_labels)
    model_training.evaluate_plain_text(test_data, test_labels)


def prepare_data(data, labels):
    data = data.reshape(data.shape[0], MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS)
    data = data.astype('float32')
    data /= 255

    labels = to_categorical(labels, NUM_CLASSES)
    return data, labels


train_mnist_model()
