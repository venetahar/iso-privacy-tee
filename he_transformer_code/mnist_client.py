from mnist_util import client_argument_parser
import numpy as np

from he_transformer_code.generic_client import perform_inference

if __name__ == "main":
    parameters, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Supplied parameters cannot be parsed", unparsed)
        exit(1)

    test_data = np.load('mnist/bob_test_data.npy')
    test_data_labels = np.load('mnist/bob_test_data_labels.npy')

    perform_inference(test_data, test_data_labels, parameters)
