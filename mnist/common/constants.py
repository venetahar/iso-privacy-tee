MNIST_WIDTH = 28
MNIST_HEIGHT = 28
NUM_CHANNELS = 1
INPUT_SHAPE = (MNIST_WIDTH, MNIST_HEIGHT, NUM_CHANNELS)

DENSE_UNITS = [128, 128]
NUM_CLASSES = 10

TRAINING_PARAMS = {
        'learning_rate': 0.001,
        'momentum': 0.9,
        'num_epochs': 15,
        'optimizer': 'Adam',
        'batch_size': 128
}

MNIST_NORM_MEAN = [0.1307]
MNIST_NORM_STD = [0.3081]
CONV_DENSE_UNITS = [100]
CONV_FILTERS = 5
KERNEL_SIZE = 5
STRIDE = 2
POOL_SIZE = 2

