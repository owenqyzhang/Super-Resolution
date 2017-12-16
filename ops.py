import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def conv(x, hidden_num=64, kernel_size=3, stride=1, w_decay=True):
    vs = tf.get_variable_scope()
    in_channels = x.get_shape()[3]
    if w_decay:
        weight_decay = tf.constant(0.05, dtype=tf.float32)
        w = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, hidden_num],
                            initializer=tf.contrib.layers.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    else:
        w = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, hidden_num],
                            initializer=tf.contrib.layers.variance_scaling_initializer())

    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    return x


def dense(x, hidden_num, is_train):
    vs = tf.get_variable_scope()
    in_channels = x.get_shape()[1]
    w = tf.get_variable('weights', [in_channels, hidden_num],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
    x = tf.matmul(x, w)
    x = batch_norm(x, is_train=is_train)
    x = tf.nn.relu(x)
    return x


def res_block(x, scope, flags, output_channel=64, stride=1):
    with tf.variable_scope(scope):
        x1 = conv(x, output_channel, stride=stride)
        x1 = batch_norm(x1, is_train=flags.is_training)
        x1 = prelu(x1)
        x1 = conv(x1, output_channel, stride=stride)
        x1 = batch_norm(x1, is_train=flags.is_training)
        x1 = tf.add(x, x1)

    return x1


def res_block_edsr(x, scope, output_channel=256, scale=1, stride=1):
    with tf.variable_scope(scope):
        x1 = conv(x, output_channel, stride=stride)
        x1 = tf.nn.relu(x1)
        x1 = conv(x1, output_channel, stride=stride)
        x1 = tf.multiply(x1, tf.constant(scale))
        x1 = tf.add(x, x1)

    return x1


def discriminator_block(x, output_channel, kernel_size, stride, scope, flags):
    with tf.variable_scope(scope):
        x1 = conv(x, kernel_size=kernel_size, hidden_num=output_channel, stride=stride)
        x1 = batch_norm(x1, flags.is_training)
        x1 = lrelu(x1, 0.2)

    return x1


def prelu(x, name='PReLU'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5

    return pos + neg


def lrelu(x, alpha=0.3, name='LeakyReLU'):
    with tf.variable_scope(name):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def batch_norm(x, is_train=True, decay=0.99, epsilon=0.001, name=''):
    shape_x = x.get_shape().as_list()
    beta = tf.get_variable('beta' + name, shape_x[-1], initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma' + name, shape_x[-1], initializer=tf.constant_initializer(1.0))
    moving_mean = tf.get_variable('moving_mean' + name, shape_x[-1],
                                  initializer=tf.constant_initializer(0.0), trainable=False)
    moving_var = tf.get_variable('moving_var' + name, shape_x[-1],
                                 initializer=tf.constant_initializer(1.0), trainable=False)

    if is_train:
        mean, var = tf.nn.moments(x, np.arange(len(shape_x) - 1), keep_dims=True)
        mean = tf.reshape(mean, [mean.shape.as_list()[-1]])
        var = tf.reshape(var, [var.shape.as_list()[-1]])

        update_moving_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
        update_moving_var = tf.assign(moving_var,
                                      moving_var * decay + shape_x[0] / (shape_x[0] - 1) * var * (1 - decay))
        update_ops = [update_moving_mean, update_moving_var]

        with tf.control_dependencies(update_ops):
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

    else:
        mean = moving_mean
        var = moving_var
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


def pixel_shuffler(x, scale=2):
    size = tf.shape(x)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = x.get_shape().as_list()[-1]

    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    input_split = tf.split(x, channel_target, axis=3)
    output = tf.concat([phase_shift(x, shape_1, shape_2) for x in input_split], axis=3)

    return output


def phase_shift(x, shape_1, shape_2):
    x = tf.reshape(x, shape_1)
    x = tf.transpose(x, [0, 1, 3, 2, 4])

    return tf.reshape(x, shape_2)


def print_configuration_op(flags):
    print('[Confiurations]:')
    a = flags.mode
    for name, value in flags.__flags.items():
        if type(value) == float:
            print('\t%s: %f' % (name, value))
        elif type(value) == int:
            print('\t%s: %d' % (name, value))
        elif type(value) == str:
            print('\t%s: %s' % (name, value))
        elif type(value) == bool:
            print('\t%s: %s' % (name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')


def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

    return psnr


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_19(inputs, scope='vgg_19', reuse=False):
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            x = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            x = slim.max_pool2d(x, [2, 2], scope='pool1')
            x = slim.repeat(x, 2, slim.conv2d, 128, 3, scope='conv2', reuse=reuse)
            x = slim.max_pool2d(x, [2, 2], scope='pool2')
            x = slim.repeat(x, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            x = slim.max_pool2d(x, [2, 2], scope='pool3')
            x = slim.repeat(x, 4, slim.conv2d, 512, 3, scope='conv4', reuse=reuse)
            x = slim.max_pool2d(x, [2, 2], scope='pool4')
            x = slim.repeat(x, 4, slim.conv2d, 512, 3, scope='conv5', reuse=reuse)
            x = slim.max_pool2d(x, [2, 2], scope='pool5')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return x, end_points


vgg_19.default_image_size = 224
