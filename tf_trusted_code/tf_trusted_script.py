import numpy as np
import subprocess
import os


def test_malaria_model(input_files_dir, predictions_dir):
    input_files = os.listdir(input_files_dir)
    for input_filename in input_files:
        input_file = input_files_dir + input_filename
        output_file_name = get_output_filename(input_filename)
        output_file = predictions_dir + output_file_name
        test_batch_size = get_num_samples_file(input_file)
        subprocess.run(['python3', 'model_run.py', '--model_file', 'alice_conv_pool_model.pb',
                        '--input_file', input_file, '--input_name', 'conv2d_input',
                        '--output_name', 'dense_1/Softmax', '--batch_size', str(test_batch_size),
                        '--output_file', output_file])


def get_output_filename(input_filename):
    common_file_prefix = input_filename.rsplit('.', 1)[0]
    return common_file_prefix + '_predictions.txt'


def get_num_samples_file(file_name):
    input_data = np.load(file_name)
    return input_data.shape[0]


test_malaria_model('malaria/batched_test_data/', 'malaria/predictions/')
