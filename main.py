from common.model_factory import CONV_MODEL_TYPE, FULLY_CONNECTED_MODEL_TYPE
from malaria.common.malaria_training import evaluate_saved_model, train_malaria_model
from mnist.common.mnist_training import test_saved_model, train_mnist_model

MNIST_MODEL_PATH = 'mnist/models/'
MNIST_FULLY_CONNECTED_MODEL_NAME = 'alice_fc3_model'
MNIST_CONV_MODEL_NAME = 'alice_conv_model'

MALARIA_MODEL_PATH = 'malaria/models/'
MALARIA_DATA_PATH = 'malaria/data/cell_images/'
MALARIA_MODEL_NAME = 'alice_conv_pool_model'
# train_mnist_model(FULLY_CONNECTED_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_FULLY_CONNECTED_MODEL_NAME)
test_saved_model(MNIST_MODEL_PATH + MNIST_FULLY_CONNECTED_MODEL_NAME)
# tee_eval = TrustedExecutionEnvironmentEvaluation()
# tee_eval.evaluate_predictions('../../mnist/data/predictions.txt', '../../mnist/data/bob_test_labels.npy')

# train_mnist_model(CONV_MODEL_TYPE, MNIST_MODEL_PATH, MNIST_CONV_MODEL_NAME)
test_saved_model(MNIST_MODEL_PATH + MNIST_CONV_MODEL_NAME)
# tee_eval = TrustedExecutionEnvironmentEvaluation()
# tee_eval.evaluate_predictions('../../mnist/data/predictions.txt', '../../mnist/data/bob_test_labels.npy')


train_malaria_model(model_path=MALARIA_MODEL_PATH, model_name=MALARIA_MODEL_NAME, data_path=MALARIA_DATA_PATH)
evaluate_saved_model(MALARIA_MODEL_PATH + MALARIA_MODEL_NAME,
                     'malaria/data/bob_test_data_16.npy',
                     'malaria/data/bob_test_labels_16.npy')
# tee_eval = TrustedExecutionEnvironmentEvaluation()
# tee_eval.evaluate_predictions('../../malaria/data/predictions.txt', '../../malaria/data/bob_test_labels_16.npy')
