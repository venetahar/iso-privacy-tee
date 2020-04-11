import numpy as np


class DataUtils:

    @staticmethod
    def save_model(model_path, model):
        model.save(model_path)

    @staticmethod
    def save_data(data, labels):
        np.save('../data/bob_test_data', data)
        np.save('../data/bob_test_labels', labels)
