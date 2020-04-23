import subprocess


def test_malaria_model():
    subprocess.run(['python3', 'model_run.py', '--model_file', 'alice_conv_pool_model.pb',
                    '--input_file', 'bob_test_data.npy', '--input_name', 'conv2d_input',
                    '--output_name', 'dense_1/Softmax'])


test_malaria_model()
