import numpy as np

test_data = np.load('../data/bob_test_data.npy')
test_labels = np.load('../data/bob_test_labels.npy')
print(test_data[0:4].shape)

np.save('../data/bob_test_data_4.npy', test_data[0:4])
np.save('../data/bob_test_labels_4.npy', test_labels[0:4])
