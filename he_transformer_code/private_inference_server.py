import tensorflow as tf
from mnist_util import server_config_from_flags

from common.utils.data_utils import DataUtils
from he_transformer_code.private_inference import PrivateInference


class PrivateInferenceServer(PrivateInference):

    def __init__(self, test_data_path, test_data_labels_path, parameters):
        super().__init__(test_data_path, test_data_labels_path, parameters, self.parameters.enable_client)

    def obtain_predictions(self, test_data):
        model_graph = DataUtils.load_model(self.parameters.model_file)
        tf.import_graph_def(model_graph)
        x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(self.parameters.input_node)
        y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(self.parameters.output_node)

        config = server_config_from_flags(self.parameters, x_input.name)
        with tf.compat.v1.Session(config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            prediction_scores = y_output.eval(feed_dict={x_input: test_data})
        return prediction_scores
