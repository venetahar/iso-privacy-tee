import numpy as np
import pyhe_client

from he_transformer_code.private_inference import PrivateInference


class PrivateInferenceClient(PrivateInference):

    def __init__(self, test_data_path, test_data_labels_path, parameters):
        super().__init__(test_data_path, test_data_labels_path, parameters, True)
        self.num_classes = self.test_data_labels.shape[1]

    def obtain_predictions(self, test_data):
        client = pyhe_client.HESealClient(
            self.parameters.hostname,
            self.parameters.port,
            self.parameters.batch_size,
            {
                self.parameters.tensor_name: (self.parameters.encrypt_data_str, test_data)
            }),

        prediction_scores = np.array(client.get_results()).reshape(self.parameters.batch_size, self.num_classes)
        return prediction_scores

