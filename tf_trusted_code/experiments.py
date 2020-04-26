import numpy as np
from private_inference import PrivateInference


def calculate_num_correct_predictions(prediction_scores, one_hot_labels):
    predictions = prediction_scores.argmax(axis=1)
    labels = np.where(one_hot_labels == 1)[1]
    return np.sum(predictions == labels)


def run_mnist_conv_experiment():
    test_data, test_data_labels = load_mnist_test_data()
    parameters = {
        "model_file": "mnist/alice_conv_model.pb",
        "input_name": "conv2d_input",
        "output_name": "dense_1/Softmax",
        "model_name": "alice_conv_model",
        "benchmark": False
    }
    private_inference = PrivateInference(parameters)
    prediction_scores = private_inference.perform_inference(test_data)
    correct_predictions = calculate_num_correct_predictions(prediction_scores, test_data_labels)
    print('TEE: Test set: Accuracy: ({:.4f})'.format(correct_predictions / test_data_labels.shape[0]))


def load_mnist_test_data():
    test_data = np.load("mnist/bob_test_data.npy")
    test_data_labels = np.load("mnist/bob_test_labels.npy")
    return test_data, test_data_labels


def load_malaria_test_data():
    test_data = np.load("malaria/bob_test_data.npy")
    test_data_labels = np.load("malaria/bob_test_labels.npy")
    return test_data, test_data_labels

def run_mnist_fully_connected_experiment():
    test_data, test_data_labels = load_mnist_test_data()
    parameters = {
        "model_file": "mnist/alice_fc3_model.pb",
        "input_name": "flatten_input",
        "output_name": "dense_2/Softmax",
        "model_name": "alice_fc3_model",
        "benchmark": False
    }
    private_inference = PrivateInference(parameters)
    prediction_scores = private_inference.perform_inference(test_data)
    correct_predictions = calculate_num_correct_predictions(prediction_scores, test_data_labels)
    print('TEE: Test set: Accuracy: ({:.4f})'.format(correct_predictions / test_data_labels.shape[0]))


def run_malaria_conv_experiment():
    test_data, test_data_labels = load_malaria_test_data()
    parameters = {
        "model_file": "malaria/alice_conv_pool_model.pb",
        "input_name": "conv2d_input",
        "output_name": "dense_1/Softmax",
        "model_name": "alice_conv_pool_model",
        "benchmark": False
    }

    private_inference = PrivateInference(parameters)
    batch_size = 16
    index = 0
    num_samples = test_data.shape[0]
    correct_predictions = 0
    while index < num_samples:
        new_index = index + batch_size if index + batch_size < num_samples else num_samples
        prediction_scores = private_inference.perform_inference(test_data[index: new_index])
        correct_predictions += calculate_num_correct_predictions(prediction_scores, test_data_labels[index: new_index])
        index = new_index
    print('TEE: Test set: Accuracy: ({:.4f})'.format(correct_predictions / test_data_labels.shape[0]))


if __name__ == "__main__":
    # run_mnist_fully_connected_experiment()
    # run_mnist_conv_experiment()
    run_malaria_conv_experiment()
