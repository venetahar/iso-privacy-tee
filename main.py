from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

from common.model_factory import CONV_MODEL_TYPE, FULLY_CONNECTED_MODEL_TYPE
from malaria.common.malaria_training import evaluate_saved_model, train_malaria_model
from mnist.common.mnist_training import test_saved_model, train_mnist_model

MNIST_MODEL_PATH = 'mnist/models/'
MNIST_FULLY_CONNECTED_MODEL_NAME = 'alice_fc3_model'
MNIST_CONV_MODEL_NAME = 'alice_conv_model'

MALARIA_MODEL_PATH = 'malaria/models/'
MALARIA_DATA_PATH = 'malaria/data/cell_images/'
MALARIA_MODEL_NAME = 'alice_conv_pool_model'
MALARIA_TARGET_DATA_PATH_PREFIX = 'malaria/data/bob_test_'


def evaluate_mnist_fully_connected_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(FULLY_CONNECTED_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME,
                          'mnist/data/bob_test_')
    test_saved_model(MNIST_MODEL_PATH + MNIST_FULLY_CONNECTED_MODEL_NAME)


def evaluate_mnist_conv_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(CONV_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_CONV_MODEL_NAME, 'mnist/data/bob_test_')
    test_saved_model(MNIST_MODEL_PATH + MNIST_CONV_MODEL_NAME)


def evaluate_malaria_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_malaria_model(model_path=MALARIA_MODEL_PATH, model_name=MALARIA_MODEL_NAME,
                            source_data_path=MALARIA_DATA_PATH,
                            target_data_path_prefix=MALARIA_TARGET_DATA_PATH_PREFIX)
    evaluate_saved_model(MALARIA_MODEL_PATH + MALARIA_MODEL_NAME,
                         MALARIA_TARGET_DATA_PATH_PREFIX + 'data.npy',
                         MALARIA_TARGET_DATA_PATH_PREFIX + 'labels.npy')


# evaluate_mnist_fully_connected_experiment(True)
# evaluate_mnist_conv_experiment(True)
evaluate_malaria_experiment(True)
