import argparse

import numpy as np
from private_inference import PrivateInference


def calculate_num_correct_predictions(prediction_scores, one_hot_labels):
    """
    Calculates the number of correct predictions.
    :param prediction_scores: The prediction scores.
    :param one_hot_labels: The one hot labels.
    :return: the number of correct predictions.
    """
    predictions = prediction_scores.argmax(axis=1)
    labels = np.where(one_hot_labels == 1)[1]
    return np.sum(predictions == labels)


def evaluate_model(parameters, should_benchmark, test_data, test_data_labels):
    """
    Evaluates the model.
    :param parameters: The parameters.
    :param should_benchmark: Whether it should benchmark.
    :param test_data: The test data.
    :param test_data_labels: The test labels.
    :return:
    """
    private_inference = PrivateInference(parameters)
    if should_benchmark:
        private_inference.perform_inference(test_data[0:1])
    else:
        prediction_scores = private_inference.perform_inference(test_data)
        correct_predictions = calculate_num_correct_predictions(prediction_scores, test_data_labels)
        print('TEE: Test set: Accuracy: ({:.4f})'.format(correct_predictions / test_data_labels.shape[0]))


def load_mnist_test_data():
    """
    Loads the mnist test data and labels.
    :return: the mnist test data and labels.
    """
    test_data = np.load("mnist/data/bob_test_data.npy")
    test_data_labels = np.load("mnist/data/bob_test_labels.npy")
    return test_data, test_data_labels


def load_malaria_test_data():
    """
    Loads the malaria test data and labels.
    :return: the malaria test data and labels.
    """
    test_data = np.load("malaria/data/bob_test_data.npy")
    test_data_labels = np.load("malaria/data/bob_test_labels.npy")
    return test_data, test_data_labels


def run_mnist_fully_connected_experiment(should_benchmark=False):
    """
    Runs the mnist fc experiment.
    :param should_benchmark: Whether it should benchmark.
    """
    test_data, test_data_labels = load_mnist_test_data()
    parameters = {
        "model_file": "mnist/models/alice_fc3_model.pb",
        "input_name": "flatten_input",
        "output_name": "dense_2/Softmax",
        "model_name": "alice_fc3_model",
        "benchmark": should_benchmark
    }
    evaluate_model(parameters, should_benchmark, test_data, test_data_labels)


def run_mnist_conv_experiment(should_benchmark=False):
    """
    Runs the mnist conv experiment.
    :param should_benchmark: Whether it should benchmark.
    """
    test_data, test_data_labels = load_mnist_test_data()
    parameters = {
        "model_file": "mnist/models/alice_conv_model.pb",
        "input_name": "conv2d_input",
        "output_name": "dense_1/Softmax",
        "model_name": "alice_conv_model",
        "benchmark": should_benchmark
    }
    evaluate_model(parameters, should_benchmark, test_data, test_data_labels)


def run_malaria_conv_experiment(should_benchmark=False):
    """
    Runs the malaria conv experiment.
    :param should_benchmark: Whether it should benchmark.
    """
    test_data, test_data_labels = load_malaria_test_data()
    parameters = {
        "model_file": "malaria/models/alice_conv_pool_model.pb",
        "input_name": "conv2d_input",
        "output_name": "dense_1/Softmax",
        "model_name": "alice_conv_pool_model",
        "benchmark": should_benchmark
    }

    private_inference = PrivateInference(parameters)
    if should_benchmark:
        private_inference.perform_inference(test_data[0: 1])
    else:
        batch_size = 18
        index = 0
        num_samples = test_data.shape[0]
        correct_predictions = 0
        while index < num_samples:
            new_index = index + batch_size if index + batch_size < num_samples else num_samples
            prediction_scores = private_inference.perform_inference(test_data[index: new_index])
            correct_predictions += calculate_num_correct_predictions(prediction_scores,
                                                                     test_data_labels[index: new_index])
            index = new_index
        print('TEE: Test set: Accuracy: ({:.4f})'.format(correct_predictions / test_data_labels.shape[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='',
                        help='The experiment name. Can be either: mnist_fc, mnist_conv or malaria_conv')
    parser.add_argument('--benchmark', type=bool, default=False,
                        help='Whether to benchmark the experiment. Default False.')
    config = parser.parse_args()

    if config.experiment_name == 'mnist_fc':
        run_mnist_fully_connected_experiment(config.benchmark)
    elif config.experiment_name == 'mnist_conv':
        run_mnist_conv_experiment(config.benchmark)
    elif config.experiment_name == 'malaria_conv':
        run_malaria_conv_experiment(config.benchmark)
    else:
        print("Please supply a valid experiment type. Can be either: mnist_fc, mnist_conv or malaria_conv ")
