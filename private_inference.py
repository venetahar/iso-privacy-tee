import numpy as np
import tensorflow as tf
import os
import time
import tf_trusted_custom_op as tft

from model_utils import ModelUtils


class PrivateInference:

    def __init__(self, parameters):
        self.parameters = parameters

    def perform_inference(self, test_data):
        dirname = os.path.dirname(tft.__file__)
        shared_object = dirname + '/model_enclave_op.so'

        model_module = tf.load_op_library(shared_object)
        model_load = model_module.model_load_enclave
        model_predict = model_module.model_predict_enclave

        model_file = self.parameters["model_file"]
        input_name = self.parameters["input_name"]
        output_name = self.parameters["output_name"]
        benchmark = self.parameters["benchmark"]
        model_name = self.parameters["model_name"]
        num_samples = test_data.shape[0]
        print("Num samples", num_samples)
        with tf.Session():
            input_shape = ModelUtils.get_input_shape(input_name, num_samples, model_file)
            output_shape, output_type = ModelUtils.get_output_shape_and_type(output_name, num_samples, model_file)

        tflite_bytes = ModelUtils.convert_model_to_tflite(model_file, [input_name], [output_name], input_shape)

        if benchmark:
            self.benchmark_inference(input_shape, model_load, model_name, model_predict, output_shape, output_type,
                                     test_data, tflite_bytes)
        else:
            with tf.Session() as sess:
                load_node = model_load(model_name, tflite_bytes)
                load_node.run()
                prediction_scores = PrivateInference.evaluate_model(input_shape, model_name, model_predict,
                                                                    output_shape, output_type, sess, test_data)
                return prediction_scores

    @staticmethod
    def benchmark_inference(input_shape, model_load, model_name, model_predict, output_shape, output_type,
                            test_data, tflite_bytes):
        tf.reset_default_graph()
        with tf.Session() as sess:
            load_node = model_load(model_name, tflite_bytes)
            load_node.run()
            run_times = []
            for i in range(0, 20):
                print("Run", i)
                start_time = time.time()
                PrivateInference.evaluate_model(input_shape, model_name, model_predict, output_shape, output_type,
                                                sess, test_data)
                end_time = time.time()
                run_times.append(end_time - start_time)
        print("============Performance metrics: ============ ")
        print("Average evaluate model time: {}".format(np.mean(run_times)))

    @staticmethod
    def evaluate_model(input_shape, model_name, model_predict, output_shape, output_type, sess, test_data):
        placeholder = tf.placeholder(shape=input_shape, dtype=tf.float32)
        out = model_predict(model_name, placeholder, output_shape, dtype=output_type)
        return sess.run(out, feed_dict={placeholder: test_data})


