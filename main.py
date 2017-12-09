import tensorflow as tf
import os
import math
import time
import numpy as np

Flags = tf.app.flags

# System parameters
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The directory to store the summaries')
Flags.DEFINE_string('mode', 'train', 'Mode for running: train, test, or inference')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weights will be restored from the provided checkpoint')

