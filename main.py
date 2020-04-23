from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

from common.model_factory import CONV_MODEL_TYPE, FULLY_CONNECTED_MODEL_TYPE
from malaria.common.malaria_training import evaluate_saved_model, train_malaria_model
from mnist.common.mnist_training import test_saved_model, train_mnist_model
from common.utils.data_utils import DataUtils

MNIST_MODEL_PATH = 'mnist/models/'
MNIST_FULLY_CONNECTED_MODEL_NAME = 'alice_fc3_model'
MNIST_CONV_MODEL_NAME = 'alice_conv_model'

MALARIA_MODEL_PATH = 'malaria/models/'
MALARIA_DATA_PATH = 'malaria/data/cell_images/'
MALARIA_MODEL_NAME = 'alice_conv_pool_model'
MALARIA_TARGET_DATA_PATH_PREFIX = 'malaria/data/bob_test_'
MALARIA_BATCHED_TEST_DATA_DIR = 'tf_trusted_code/malaria/batched_test_data/'
MALARIA_BATCHED_TEST_DATA_LABELS_DIR = 'tf_trusted_code/malaria/batched_test_labels/'
MALARIA_BATCHED_TEST_DATA_FILE_PREFIX = 'bob_test_'


def run_mnist_fully_connected_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(FULLY_CONNECTED_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME,
                          'mnist/data/bob_test_')
    test_saved_model(MNIST_MODEL_PATH + MNIST_FULLY_CONNECTED_MODEL_NAME)
    # tee_eval = TrustedExecutionEnvironmentEvaluation()
    # tee_eval.evaluate_predictions('../../mnist/data/predictions.txt', '../../mnist/data/bob_test_labels.npy')


def run_mnist_conv_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(CONV_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_CONV_MODEL_NAME, 'mnist/data/bob_test_')
    test_saved_model(MNIST_MODEL_PATH + MNIST_CONV_MODEL_NAME)
    # tee_eval = TrustedExecutionEnvironmentEvaluation()
    # tee_eval.evaluate_predictions('../../mnist/data/predictions.txt', '../../mnist/data/bob_test_labels.npy')


def run_malaria_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_malaria_model(model_path=MALARIA_MODEL_PATH, model_name=MALARIA_MODEL_NAME,
                            source_data_path=MALARIA_DATA_PATH,
                            target_data_path_prefix=MALARIA_TARGET_DATA_PATH_PREFIX)
    evaluate_saved_model(MALARIA_MODEL_PATH + MALARIA_MODEL_NAME,
                         MALARIA_TARGET_DATA_PATH_PREFIX + 'data.npy',
                         MALARIA_TARGET_DATA_PATH_PREFIX + 'labels.npy')
    DataUtils.batch_data(MALARIA_TARGET_DATA_PATH_PREFIX + 'data.npy',
                         MALARIA_TARGET_DATA_PATH_PREFIX + 'labels.npy', 16,
                         MALARIA_BATCHED_TEST_DATA_DIR, MALARIA_BATCHED_TEST_DATA_LABELS_DIR,
                         MALARIA_BATCHED_TEST_DATA_FILE_PREFIX)
    # tee_eval = TrustedExecutionEnvironmentEvaluation()
    # tee_eval.evaluate_predictions('../../malaria/data/predictions.txt', '../../malaria/data/bob_test_labels_16.npy')

run_malaria_experiment(False)
