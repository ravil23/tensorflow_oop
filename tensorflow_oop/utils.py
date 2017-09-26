"""
Helpful functions.
"""

import tensorflow as tf

def devices_list():
    """List of available devices.

    Return:
        result -- list of pairs (name, physical_device_desc)

    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [(x.name, x.physical_device_desc) for x in local_device_protos]


def checkpoints_list(log_dir):
    """List of all model checkpoint paths.

    Arguments:
        log_dir -- path to logging directory

    Return:
        result -- list of checkpoint paths

    """
    checkpoint_state = tf.train.get_checkpoint_state(log_dir)
    if checkpoint_state is not None:
        return [checkpoint for checkpoint in checkpoint_state.all_model_checkpoint_paths]
    else:
        return []


def nodes_list(graph=None):
    """List of graph nodes.

    Arguments:
        graph -- tensorflow graph (optional)

    Return:
        result -- list of node names

    """
    if graph is None:
        graph = tf.get_default_graph()
    return [node.name for node in graph.as_graph_def().node]


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
    from tensorflow import graph_util
    input_graph_def = sess.graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names,
        variable_names_whitelist=variable_names_whitelist,
        variable_names_blacklist=variable_names_blacklist)
    with tf.gfile.GFile(filename, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def freeze_checkpoint(filename, checkpoint_path,
                      output_node_names,
                      variable_names_whitelist=None,
                      variable_names_blacklist=None):
    """Save session graph as binary file.

    Arguments:
        filename -- output path
        checkpoint_path -- path to checkpoint file
        output_node_names -- list of name strings for the result nodes of the graph
        variable_names_whitelist -- the set of variable names to convert
                                    (by default, all variables are converted)
        variable_names_blacklist -- the set of variable names to omit converting to constants

    """
    # Import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta', clear_devices=True)

    # Start session to restore the graph weights and freeze variables to constants
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        freeze_session(filename, sess,
                       output_node_names,
                       variable_names_whitelist=variable_names_whitelist,
                       variable_names_blacklist=variable_names_blacklist)
