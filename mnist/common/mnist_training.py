from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from common.constants import DENSE_UNITS, NUM_CLASSES, TRAINING_PARAMS, MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS, \
    KERNEL_SIZE, STRIDE, CONV_DENSE_UNITS, POOL_SIZE, CONV_FILTERS, INPUT_SHAPE, CONV_MODEL_TYPE, FC_MODEL_TYPE
from common.model_training import ModelTraining
from common.utils.data_utils import DataUtils
from common.utils.tee_evaluation import TrustedExecutionEnvironmentEvaluation
from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel


def train_mnist_model(model_type=FC_MODEL_TYPE):
    """
    Trains a MNIST model and saves the model graph.
    :param model_type: The model type to use.
    """
    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
    training_data, training_labels = preprocess(training_data, training_labels)
    test_data, test_labels = preprocess(test_data, test_labels)

    model = get_model(model_type)
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train(training_data, training_labels)
    model_training.evaluate_plain_text(test_data, test_labels)
    DataUtils.save_model(model_path='../models/alice_conv_model', model=model)
    DataUtils.save_data(test_data, test_labels)
    DataUtils.save_graph(model, '../models/alice_model_dir')


def preprocess(data, labels):
    """
    Pre-processes the data.
    :param data: The data to pre-process.
    :param labels: The labels.
    :return: Pre-processed data and labels.
    """
    data = data.reshape(data.shape[0], MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS)
    data = data.astype('float32')
    data /= 255

    labels = to_categorical(labels, NUM_CLASSES)
    return data, labels


def get_model(model_type='FC'):
    """
    Returns the model based on the model type. By default it returns the fully connected model.
    :param model_type: The model type.
    :return: An initialised model.
    """
    if model_type == CONV_MODEL_TYPE:
        return ConvModel(kernel_size=KERNEL_SIZE, stride=STRIDE, channels=CONV_FILTERS,
                         pool_size=POOL_SIZE, dense_units=CONV_DENSE_UNITS, num_classes=NUM_CLASSES,
                         input_shape=INPUT_SHAPE).model
    else:
        return FullyConnectedModel(dense_units=DENSE_UNITS, num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE).model


def test_saved_model():
    (_, _), (test_data, test_labels) = mnist.load_data()
    test_data, test_labels = preprocess(test_data, test_labels)

    new_model = load_model('../models/alice_conv_model.h5')
    new_model.evaluate(test_data, test_labels)


# train_mnist_model(CONV_MODEL_TYPE)
test_saved_model()
tee_eval = TrustedExecutionEnvironmentEvaluation()
tee_eval.evaluate_predictions('../../mnist/data/predictions.txt', '../../mnist/data/bob_test_labels.npy')
