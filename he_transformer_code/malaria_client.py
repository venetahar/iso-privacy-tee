from mnist_util import client_argument_parser

from he_transformer_code.private_inference_client import PrivateInferenceClient

if __name__ == "main":
    parameters, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Supplied parameters cannot be parsed", unparsed)
        exit(1)

    malaria_private_inference_client = PrivateInferenceClient("malaria/bob_test_data.npy",
                                                              "malaria/bob_test_data_labels.npy",
                                                              parameters)
    malaria_private_inference_client.perform_inference()
