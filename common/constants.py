MNIST_WIDTH = 28
MNIST_HEIGHT = 28
NUM_CHANNELS = 1

DENSE_UNITS = [128, 128]
NUM_CLASSES = 10

TRAINING_PARAMS = {
        'learning_rate': 0.001,
        'momentum': 0.9,
        'num_epochs': 10,
        'optimizer': 'Adam',
        'batch_size': 128
}
