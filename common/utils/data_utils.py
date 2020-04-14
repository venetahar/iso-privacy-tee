import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class DataUtils:

    @staticmethod
    def save_model(model_path, model):
        """
        Saves the model in a default keras h5 format.
        :param model_path: The model path.
        :param model: The model.
        """
        model.save(model_path)

    @staticmethod
    def save_graph(model, model_path, model_name='alice_model.pb'):
        """
        Saves the model graph as .pb.
        :param model: The model.
        :param model_path: The model path.
        :param model_name: The model name. By default it's alice_model.pb.
        """
        frozen_graph = DataUtils.freeze_graph(model=model, session=K.get_session())
        tf.io.write_graph(frozen_graph, model_path, model_name, as_text=False)

    @staticmethod
    def freeze_graph(model, session):
        """
        Freezes the model graph. This code is based on this answer:
        https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
        :param model: The model.
        :param session: The session.
        :return: The frozen graph.
        """
        output_names = [output.op.name for output in model.outputs]
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(variable.op.name for variable in tf.global_variables()).difference([]))
            output_names += [variable.op.name for variable in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            for node in input_graph_def.node:
                node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                        output_names, freeze_var_names)
            return frozen_graph

    @staticmethod
    def save_data(data, labels, path_prefix='../data/bob_test_', ):
        """
        Saves the data in an .npy format.
        :param data: The data.
        :param labels: The labels.
        :param path_prefix: The path prefix. By default: '../data/bob_test_'
        """
        np.save(path_prefix + 'data', data)
        np.save(path_prefix + 'labels', labels)

    @staticmethod
    def sava_data_generator(data_generator):
        num_steps = 1
        steps_done = 0
        all_data = None
        all_labels = None
        while steps_done < num_steps:
            data, labels = next(data_generator)
            all_data = data if all_data is None else np.concatenate((all_data, data), axis=0)
            all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)
            steps_done += 1

        DataUtils.save_data(all_data, all_labels)

