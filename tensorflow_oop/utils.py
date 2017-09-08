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
