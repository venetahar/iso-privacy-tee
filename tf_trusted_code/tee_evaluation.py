import numpy as np
import os


class TeeEvaluation:

    @staticmethod
    def analyse_results(prediction_files_dir, labels_files_dir):
        total_predictions = 0
        total_correct_predictions = 0
        prediction_files = os.listdir(prediction_files_dir)
        for prediction_filename in prediction_files:
            prediction_file = prediction_files_dir + prediction_filename
            prediction_scores = np.loadtxt(prediction_file)
            predictions = prediction_scores.argmax(axis=1)
            labels = TeeEvaluation.get_corresponding_labels(labels_files_dir, prediction_filename)
            total_predictions += predictions.shape[0]
            total_correct_predictions += np.sum(predictions == labels)
        print('TEE: Test set: Accuracy: ({:.4f})'.format(total_correct_predictions / total_predictions))

    @staticmethod
    def get_corresponding_labels(labels_files_dir, pred_filename):
        common_prefix = pred_filename.rsplit('_', 1)[0]
        labels_file = labels_files_dir + common_prefix + '_labels.npy'
        labels_one_hot = np.load(labels_file)

        labels = np.where(labels_one_hot == 1)[1]
        return labels


TeeEvaluation.analyse_results('malaria/predictions/', 'malaria/batched_test_labels/')
