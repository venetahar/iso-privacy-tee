from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD


class ModelTraining:

    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters
        if self.parameters['optimizer'] == 'Adam':
            optimizer = Adam(self.parameters['learning_rate'])
        else:
            optimizer = SGD(self.parameters['learning_rate'], self.parameters['momentum'], nesterov=True)
        self.model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    def train(self, data, labels):
        self.model.fit(data, labels, batch_size=self.parameters['batch_size'], epochs=self.parameters['num_epochs'],
                       verbose=1, shuffle=True)
        print(self.model.summary())

    def evaluate_plain_text(self, test_data, test_labels):
        metrics = self.model.evaluate(test_data, test_labels, verbose=0)
        print(print('Test set: Loss: ({:.4f}%) Accuracy: ({:.4f}%)'.format(metrics[0], metrics[1])))
