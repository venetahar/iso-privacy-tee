from tensorflow import gfile
from tensorflow.python.client import timeline
import tensorflow as tf


def get_output_shape_and_type(output_name, batch_size, model_file):
    graph_def = load_pb_model(model_file)
    tf.graph_util.import_graph_def(graph_def)
    print('import/' + output_name + ":0")
    output = tf.get_default_session().graph.get_tensor_by_name('import/' + output_name + ":0")
    shape = list(output.get_shape())
    shape[0] = batch_size

    return shape, output.dtype


def load_pb_model(model_file):
    with gfile.GFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def get_input_shape(input_name, batch_size, model_file):
    graph_def = load_pb_model(model_file)
    tf.graph_util.import_graph_def(graph_def)
    input_node = tf.get_default_session().graph.get_tensor_by_name('import/' + input_name + ":0")
    try:
        shape = list(input_node.get_shape())
        shape[0] = batch_size
        return shape
    except ValueError:
        print("Error: Can't read shape from input try setting --input_shape instead")
        exit()


def convert_model_to_tflite(graph_def_file, input_arrays, output_arrays, input_shape):
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays,
                                                          output_arrays,
                                                          input_shapes={input_arrays[0]: input_shape})
    tflite_model = converter.convert()

    return bytes(tflite_model)


def save_to_tensorboard(i, sess, run_metadata):
    writer = tf.summary.FileWriter("/tmp/tensorboard/run" + str(i), sess.graph)

    session_tag = "prediction" + str(i)
    writer.add_run_metadata(run_metadata, session_tag)
    writer.close()

    tracer = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = tracer.generate_chrome_trace_format()
    with open('{}/{}.ctr'.format("/tmp/tensorboard", session_tag), 'w') as f:
        f.write(chrome_trace)
