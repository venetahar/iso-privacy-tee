from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from common.constants import DENSE_UNITS, NUM_CLASSES, TRAINING_PARAMS, MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS, \
    KERNEL_SIZE, STRIDE, CONV_DENSE_UNITS, POOL_SIZE, CONV_FILTERS
from common.model_training import ModelTraining
from common.utils.data_utils import DataUtils
from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel


def train_mnist_model(model_type='FC'):
    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
    training_data, training_labels = prepare_data(training_data, training_labels)
    test_data, test_labels = prepare_data(test_data, test_labels)

    model = get_model(model_type)
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train(training_data, training_labels)
    model_training.evaluate_plain_text(test_data, test_labels)
    DataUtils.save_model(model_path='../models/alice_conv_model', model=model)
    DataUtils.save_data(test_data, test_labels)


def prepare_data(data, labels):
    data = data.reshape(data.shape[0], MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS)
    data = data.astype('float32')
    data /= 255

    labels = to_categorical(labels, NUM_CLASSES)
    return data, labels


def get_model(model_type='FC'):
    if model_type == 'Conv':
        return ConvModel(kernel_size=KERNEL_SIZE, stride=STRIDE, channels=CONV_FILTERS,
                         pool_size=POOL_SIZE, dense_units=CONV_DENSE_UNITS, num_classes=NUM_CLASSES).model
    else:
        return FullyConnectedModel(DENSE_UNITS, NUM_CLASSES).model


def test_saved_model():
    (_, _), (test_data, test_labels) = mnist.load_data()
    test_data, test_labels = prepare_data(test_data, test_labels)

    new_model = load_model('../models/alice_conv_model')
    new_model.evaluate(test_data, test_labels)


train_mnist_model('Conv')
test_saved_model()
