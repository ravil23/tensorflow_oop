"""
Helpful functions.
"""

import tensorflow as tf


class TFUtils(object):

    """
    Class for grouping static utility functions.
    """

    @static_method
    def devices_list():
        """List of available devices.

        Return:
            result      List of device information pairs (name, physical_device_desc).

        """
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [(x.name, x.physical_device_desc) for x in local_device_protos]

    @static_method
    def checkpoints_list(log_dir):
        """List of all model checkpoint paths.

        Arguments:
            log_dir     Path to logging directory.

        Return:
            result      List of checkpoint paths.

        """
        checkpoint_state = tf.train.get_checkpoint_state(log_dir)
        if checkpoint_state is not None:
            return [checkpoint for checkpoint in checkpoint_state.all_model_checkpoint_paths]
        else:
            return []

    @static_method
    def nodes_list(graph=None):
        """List of graph nodes.

        Arguments:
            graph       TensorFlow graph (optional).

        Return:
            result      List of node names.

        """
        if graph is None:
            graph = tf.get_default_graph()
        return [node.name for node in graph.as_graph_def().node]

    @static_method
    def freeze_session(filename, sess, output_node_names, whitelist=None, blacklist=None):
        """Save session graph as binary file.

        Arguments:
            filename           Output path.
            sess               Session with variables and graph.
            output_node_names  List of names for the result nodes of the graph.
            whitelist          The set of variable names to convert,
                               by default, all variables are converted.
            blacklist          The set of variable names to omit converting to constants.

        """

        # Convert variables to constant
        from tensorflow import graph_util
        input_graph_def = sess.graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names,
            variable_names_whitelist=whitelist,
            variable_names_blacklist=blacklist)

        # Write to file
        with tf.gfile.GFile(filename, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    @static_method
    def freeze_checkpoint(filename, checkpoint_path,
                          output_node_names,
                          whitelist=None,
                          blacklist=None):
        """Save session graph as binary file.

        Arguments:
            filename           Output path.
            checkpoint_path    Path to checkpoint file.
            output_node_names  List of names for the result nodes of the graph.
            whitelist          The set of variable names to convert,
                               by default, all variables are converted.
            blacklist          The set of variable names to omit converting to constants.

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
