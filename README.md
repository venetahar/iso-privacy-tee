# iso-privacy-tee
Hosts experiments for Private Inference on Trusted Execution Environments

This repository contains the necessary logic to train and save a Keras model which is later used for evaluation.

In order to perform the evaluation on trusted hardware you can use the following library: 

https://github.com/capeprivacy/tf-trusted

You can follow the instructions in the README file for setting that project up. It's important to note that you need an older version of bazel for
it to build correctly  (I am using 0.24.1 and it seems to work).

To perform private inference do the following steps:
:q!

1. Find a trained MNIST model in the following directory: ```mnist/models/alice_model_dir/alice_model.pb```
2. The test data is saved here: ```mnist/data/bob_test_data.npy```
3. Copy both the model and the data to ```tf-trusted/tf-trusted-custom-op```
4. Go to ```tf-trusted/tf-trusted-custom-op``` and run the following command.
```python3 model_run.py --model_file alice_model.pb --input_file bob_test_data.npy --input_name conv2d_input --output_name dense_1/Softmax --batch_size=10000```
5. To evaluate the accuracy copy the predictions produced to ```mnist/data/``` and run the ```mnist/common/mnist_training.py``` file. 