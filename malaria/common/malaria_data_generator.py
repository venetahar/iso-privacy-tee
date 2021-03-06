from tensorflow.keras.preprocessing.image import ImageDataGenerator

from malaria.common.constants import MALARIA_NORM_MEAN, MALARIA_NORM_STD


class MalariaDataGenerator:
    """
    Creates a train and test malaria generator.
    """

    def __init__(self, data_path, parameters):
        """
        Creates a MalariaDataGenerator.
        :param data_path: The data path.
        :param parameters: THe parameters.
        """

        self.parameters = parameters

        self.data_generator = ImageDataGenerator(
            validation_split=self.parameters['test_split'],
            dtype='float32',
            preprocessing_function=self.normalize
        )

        self.train_data_generator = self.data_generator.flow_from_directory(
            data_path,
            target_size=self.parameters['target_size'],
            batch_size=self.parameters['batch_size'],
            class_mode='categorical',
            subset='training'
        )

        self.test_data_generator = self.data_generator.flow_from_directory(
            data_path,
            target_size=self.parameters['target_size'],
            batch_size=self.parameters['test_batch_size'],
            class_mode='categorical',
            subset='validation'
        )

    def normalize(self, data):
        """
        Normalizes the data.
        :param data: The data.
        :return: The normalized data.
        """
        return (data/255.0 - MALARIA_NORM_MEAN) / MALARIA_NORM_STD


