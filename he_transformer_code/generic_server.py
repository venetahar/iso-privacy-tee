import numpy as np
import tensorflow as tf
import ngraph_bridge

from mnist_util import (
    server_config_from_flags,
    load_pb_file,
    print_nodes,
)


def perform_inference(test_data, test_labels, parameters):
    tf.import_graph_def(load_pb_file(parameters.model_file))
    print("Loaded model")
    print_nodes()

    model_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
        parameters.input_node)
    model_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
        parameters.output_node)

    config = server_config_from_flags(parameters, model_input.name)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        prediction_scores = model_output.eval(feed_dict={model_input: test_data})

    if not parameters.enable_client:
        correct_predictions = calculate_num_correct_predictions(prediction_scores, test_labels)
        print('HE-Transformer: Test set: Accuracy: ({:.4f})'.format(correct_predictions / parameters.batch_size))


def calculate_num_correct_predictions(prediction_scores, one_hot_labels):
    predictions = prediction_scores.argmax(axis=1)
    labels = np.where(one_hot_labels == 1)[1]
    return np.sum(predictions == labels)

