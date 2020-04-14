from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD


class ModelTraining:
    """
    Responsible for training a Keras model.
    """

    def __init__(self, model, parameters):
        """
        Creates a ModelTraining.
        :param model: The model used for training.
        :param parameters: The training parameters.
        """
        self.model = model
        self.parameters = parameters
        if self.parameters['optimizer'] == 'Adam':
            optimizer = Adam(self.parameters['learning_rate'])
        else:
            optimizer = SGD(self.parameters['learning_rate'], self.parameters['momentum'], nesterov=True)
        self.model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    def train(self, data, labels):
        """
        Trains the model.
        :param data: The data used for training.
        :param labels: The labels.
        """
        self.model.fit(data, labels, batch_size=self.parameters['batch_size'], epochs=self.parameters['num_epochs'],
                       verbose=1, shuffle=True)
        print(self.model.summary())

    def train_generator(self, training_data_generator):
        """
        Performs training using a training data generator.
        :param training_data_generator: The training data generator.
        """
        num_steps = len(training_data_generator)
        self.model.fit_generator(training_data_generator, steps_per_epoch=num_steps,
                                 epochs=self.parameters['num_epochs'], verbose=1, shuffle=True)
        print(self.model.summary())

    def evaluate_plain_text(self, test_data, test_labels):
        """
        Evaluates the model in plain text.
        :param test_data: The test data.
        :param test_labels: The test labels.
        """
        metrics = self.model.evaluate(test_data, test_labels, verbose=0)
        print('Test set: Loss: ({:.4f}%) Accuracy: ({:.4f}%)'.format(metrics[0], metrics[1]))

    def evaluate_generator(self, test_data_generator):
        """
        Evaluates the model using a test data generator.
        :param test_data_generator: The test data generator.
        """
        num_steps = len(test_data_generator)
        metrics = self.model.evaluate_generator(test_data_generator, steps=num_steps)
        print('Test set: Loss: ({:.4f}%) Accuracy: ({:.4f}%)'.format(metrics[0], metrics[1]))
