import tensorflow as tf
from ops import *


def generator_SRResNet(gen_inputs, gen_output_channels, reuse=False, flags=None):
    if flags is None:
        raise ValueError('No FLAGS is provided for generator')

    with tf.variable_scope('generator_unit', reuse=reuse):
        with tf.variable_scope('input_stage'):
            x1 = conv(gen_inputs, hidden_num=64, kernel_size=9, stride=1)
            x1 = prelu(x1)

        stage1 = x1

        for i in range(1, flags.num_resblocks + 1, 1):
            name_scope = 'resblock' + str(i)
            x1 = res_block(x1, 64, 1, name_scope, flags)

        with tf.variable_scope('resblock_output'):
            x1 = conv(x1)
            x1 = batch_norm(x1, flags.is_training)

        x1 = tf.add(x1, stage1)

        with tf.variable_scope('sub_pixel_conv_stage1'):
            x1 = conv(x1, hidden_num=256)
            x1 = pixel_shuffler(x1, scale=2)
            x1 = prelu(x1)

        with tf.variable_scope('output_stage'):
            x1 = conv(x1, gen_output_channels, kernel_size=9)

    return x1
