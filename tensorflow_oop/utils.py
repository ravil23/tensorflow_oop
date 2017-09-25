"""
Helpful functions.
"""

import tensorflow as tf

def devices_list():
    """List of available devices.

    Return:
        result -- tuple of pairs (name, physical_device_desc)

    """
    local_device_protos = tensorflow.python.client.device_lib.list_local_devices()
    return [(x.name, x.physical_device_desc) for x in local_device_protos]


def checkpoints_list(log_dir):
    """List of all model checkpoint paths.

    Arguments:
        log_dir -- path to logging directory

    Return:
        result -- tuple of checkpoint paths

    """
    checkpoint_state = tf.train.get_checkpoint_state(log_dir)
    return checkpoint_state.all_model_checkpoint_paths


def freeze_session(filename, sess,
                   output_node_names,
                   variable_names_whitelist=None,
                   variable_names_blacklist=None):
    """Save session graph as binary file.

    Arguments:
        filename -- output path
        sess -- session with variables and graph
        output_node_names -- list of name strings for the result nodes of the graph
        variable_names_whitelist -- the set of variable names to convert
                                    (by default, all variables are converted)
        variable_names_blacklist -- the set of variable names to omit converting to constants

    """
    input_graph_def = sess.graph.as_graph_def()
    output_graph_def = tensorflow.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names,
        variable_names_whitelist=None,
        variable_names_blacklist=None)
    with tf.gfile.GFile(filename, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def freeze_checkpoint(filename, checkpoint_path,
                      output_node_names,
                      variable_names_whitelist=None,
                      variable_names_blacklist=None):
    """Save session graph as binary file.

    Arguments:
        filename -- output path
        sess -- session with variables and graph
        output_node_names -- list of name strings for the result nodes of the graph
        variable_names_whitelist -- the set of variable names to convert
                                    (by default, all variables are converted)
        variable_names_blacklist -- the set of variable names to omit converting to constants

    """
    input_graph_def = sess.graph.as_graph_def()
    output_graph_def = tensorflow.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names,
        variable_names_whitelist=None,
        variable_names_blacklist=None)
    with tf.gfile.GFile(filename, "wb") as f:
        f.write(output_graph_def.SerializeToString())
