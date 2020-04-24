import numpy as np


class PrivateInference:

    def __init__(self, test_data_path, test_data_labels_path, parameters, should_calculate_accuracy):
        self.test_data = np.load(test_data_path)
        self.test_data_labels = np.load(test_data_labels_path)
        self.parameters = parameters
        self.should_calculate_accuracy = should_calculate_accuracy

    def perform_inference(self):
        start_index = 0
        num_samples = self.test_data.shape[0]
        batch_size = self.parameters.batch_size

        total_correct_predictions = 0
        while start_index < num_samples:
            new_index = start_index + batch_size if start_index + batch_size < num_samples else num_samples

            if self.should_calculate_accuracy:
                prediction_scores = self.obtain_predictions(self.test_data[start_index:new_index])
                total_correct_predictions += self.calculate_num_correct_predictions(
                    prediction_scores, self.test_data_labels[start_index:new_index])
            start_index = new_index
        if self.should_calculate_accuracy:
            print('HE-Transformer: Test set: Accuracy: ({:.4f})'.format(total_correct_predictions / num_samples))

    def obtain_predictions(self, test_data):
        print("obtain_predictions needs to be implemented")

    @staticmethod
    def calculate_num_correct_predictions(prediction_scores, one_hot_labels):
        predictions = prediction_scores.argmax(axis=1)
        labels = np.where(one_hot_labels == 1)[1]
        return np.sum(predictions == labels)
