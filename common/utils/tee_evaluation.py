import numpy as np


class TrustedExecutionEnvironmentEvaluation:

    @staticmethod
    def evaluate_predictions(prediction_file_location, labels_file_location):
        target = np.load(labels_file_location)
        labels = np.where(target == 1)[1]
        prediction_scores = np.loadtxt(prediction_file_location)
        predictions = prediction_scores.argmax(axis=1)
        print(predictions)
        correct_predictions = np.sum(predictions == labels)
        print('TEE: Test set: Accuracy: ({:.4f})'.format(correct_predictions / len(target)))

