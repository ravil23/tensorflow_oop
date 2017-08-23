"""
Helpful functions.
"""

import tensorflow as tf

def devices_list():
    """List of available devices."""
    local_device_protos = tf.python.client.device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def checkpoints_list(log_dir):
    """List of all model checkpoint paths."""
    checkpoint_state = tf.train.get_checkpoint_state(log_dir)
    return checkpoint_state.all_model_checkpoint_paths
