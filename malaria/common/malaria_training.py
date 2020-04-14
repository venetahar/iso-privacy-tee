import numpy as np
from tensorflow.keras.models import load_model

from common.model_training import ModelTraining
from common.utils.data_utils import DataUtils
from common.utils.tee_evaluation import TrustedExecutionEnvironmentEvaluation
from malaria.common.constants import KERNEL_SIZES, POOL_SIZES, STRIDE, CHANNELS, DENSE_UNITS, NUM_CLASSES, \
    MALARIA_INPUT_SHAPE, TRAINING_PARAMS, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE, TEST_PERCENTAGE, IMG_RESIZE
from malaria.common.conv_model import ConvModel
from malaria.common.malaria_data_generator import MalariaDataGenerator


def train_malaria_model():
    """
    Trains a Malaria model and saves the model graph.
    """
    malaria_data_generator = MalariaDataGenerator('../data/cell_images',
                                                  parameters={
                                                      'test_batch_size': TEST_BATCH_SIZE,
                                                      'batch_size': TRAIN_BATCH_SIZE,
                                                      'test_split': TEST_PERCENTAGE,
                                                      'target_size': IMG_RESIZE
                                                  })

    model = ConvModel(kernel_sizes=KERNEL_SIZES, stride=STRIDE, channels=CHANNELS,
                      pool_sizes=POOL_SIZES, dense_units=DENSE_UNITS, num_classes=NUM_CLASSES,
                      input_shape=MALARIA_INPUT_SHAPE).model
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train_generator(malaria_data_generator.train_data_generator)
    model_training.evaluate_generator(malaria_data_generator.test_data_generator)
    DataUtils.sava_data_generator(malaria_data_generator.test_data_generator)
    DataUtils.save_model(model_path='../models/alice_conv_model.h5', model=model)
    DataUtils.save_graph(model, '../models/alice_model_dir')


def evaluate_saved_model():
    test_data = np.load('../data/bob_test_data_16.npy')
    test_labels = np.load('../data/bob_test_labels_16.npy')
    new_model = load_model('../models/alice_conv_model.h5')
    pred = new_model.predict(test_data)
    print(pred)
    pred = pred.argmax(axis=1)
    print(pred)
    new_model.evaluate(test_data, test_labels)


# train_malaria_model()
evaluate_saved_model()
tee_eval = TrustedExecutionEnvironmentEvaluation()
tee_eval.evaluate_predictions('../../malaria/data/predictions.txt', '../../malaria/data/bob_test_labels_16.npy')
