IMG_RESIZE = (32, 32)

MALARIA_NORM_MEAN = [0.5, 0.5, 0.5]
MALARIA_NORM_STD = [0.5, 0.5, 0.5]

TEST_PERCENTAGE = 0.1
# Chosen so that they are exact divisors of the num samples
TRAIN_BATCH_SIZE = 52
TEST_BATCH_SIZE = 54

TRAINING_PARAMS = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'num_epochs': 20,
    'optimizer': 'Adam',
    'test_batch_size': TEST_BATCH_SIZE,
    'batch_size': TRAIN_BATCH_SIZE,
}

MALARIA_INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 2
KERNEL_SIZES = [5, 5]
CHANNELS = [36, 36]
POOL_SIZES = [2, 2]
DENSE_UNITS = [72]
STRIDE = 1
