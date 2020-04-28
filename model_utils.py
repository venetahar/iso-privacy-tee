from tensorflow import gfile
from tensorflow.python.client import timeline
import tensorflow as tf


class ModelUtils:
    """
    A utility class for performing several model operations. The code is based on the example here:
    https://github.com/capeprivacy/tf-trusted/blob/master/tf_trusted_custom_op/model_run.py
    """

    @staticmethod
    def get_output_shape_and_type(output_name, batch_size, model_file):
        """
        Returns the ouput shape and type of a model.
        :param output_name: The output node name.
        :param batch_size: The batch size.
        :param model_file: The model file.
        :return: The shape and the type of the output node.
        """
        graph_def = ModelUtils.load_pb_model(model_file)
        tf.graph_util.import_graph_def(graph_def)
        print('import/' + output_name + ":0")
        output = tf.get_default_session().graph.get_tensor_by_name('import/' + output_name + ":0")
        shape = list(output.get_shape())
        shape[0] = batch_size

        return shape, output.dtype

    @staticmethod
    def load_pb_model(model_file):
        """
        Loads a model as a graph.
        :param model_file: The model file.
        :return: Graph def created from the model.
        """
        with gfile.GFile(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    @staticmethod
    def get_input_shape(input_name, batch_size, model_file):
        """
        Returns the shape of the input node of a model.
        :param input_name: The input node name.
        :param batch_size: The batch size.
        :param model_file: The model file.
        :return: The shape of the input node of a model.
        """
        graph_def = ModelUtils.load_pb_model(model_file)
        tf.graph_util.import_graph_def(graph_def)
        input_node = tf.get_default_session().graph.get_tensor_by_name('import/' + input_name + ":0")

        shape = list(input_node.get_shape())
        shape[0] = batch_size
        return shape

    @staticmethod
    def convert_model_to_tflite(graph_def_file, input_arrays, output_arrays, input_shape):
        """
        Coverts a model to TF-Lite.
        :param graph_def_file: The graph def of the model.
        :param input_arrays: The inputs.
        :param output_arrays: The outputs.
        :param input_shape: The input shape.
        :return:
        """
        converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays,
                                                              output_arrays,
                                                              input_shapes={input_arrays[0]: input_shape})
        tflite_model = converter.convert()

        return bytes(tflite_model)

    @staticmethod
    def save_to_tensorboard(i, sess, run_metadata):
        writer = tf.summary.FileWriter("/tmp/tensorboard/run" + str(i), sess.graph)

        session_tag = "prediction" + str(i)
        writer.add_run_metadata(run_metadata, session_tag)
        writer.close()

        tracer = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = tracer.generate_chrome_trace_format()
        with open('{}/{}.ctr'.format("/tmp/tensorboard", session_tag), 'w') as f:
            f.write(chrome_trace)
