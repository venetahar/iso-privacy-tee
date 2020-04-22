import numpy as np
from tensorflow.keras.models import load_model

from common.model_factory import ModelFactory, CONV_POOL_MODEL_TYPE
from common.model_training import ModelTraining
from common.utils.data_utils import DataUtils
from malaria.common.constants import NUM_CLASSES, \
    MALARIA_INPUT_SHAPE, TRAINING_PARAMS, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE, TEST_PERCENTAGE, IMG_RESIZE
from malaria.common.malaria_data_generator import MalariaDataGenerator


def train_malaria_model(model_path, model_name, source_data_path, target_data_path_prefix):
    """
    Trains a Malaria model and saves the model graph.
    """
    malaria_data_generator = MalariaDataGenerator(source_data_path,
                                                  parameters={
                                                      'test_batch_size': TEST_BATCH_SIZE,
                                                      'batch_size': TRAIN_BATCH_SIZE,
                                                      'test_split': TEST_PERCENTAGE,
                                                      'target_size': IMG_RESIZE
                                                  })

    model = ModelFactory.create_model(CONV_POOL_MODEL_TYPE, MALARIA_INPUT_SHAPE, NUM_CLASSES)
    model_training = ModelTraining(model, TRAINING_PARAMS)
    model_training.train_generator(malaria_data_generator.train_data_generator)
    model_training.evaluate_generator(malaria_data_generator.test_data_generator)
    DataUtils.sava_data_generator(malaria_data_generator.test_data_generator, target_data_path_prefix)
    DataUtils.save_model(model_path=model_path + model_name + '.h5', model=model)
    DataUtils.save_graph(model, model_path=model_path, model_name=model_name + '.pb')


def evaluate_saved_model(model_path, test_data_path, test_labels_path):
    test_data = np.load(test_data_path)
    test_labels = np.load(test_labels_path)
    new_model = load_model(model_path + '.h5')
    pred = new_model.predict(test_data)
    print(pred)
    pred = pred.argmax(axis=1)
    print(pred)
    new_model.evaluate(test_data, test_labels)

