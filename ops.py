import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv(x, hidden_num=64, kernel_size=3, stride=1, w_decay=True, use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            if w_decay:
                weight_decay = tf.constant(0.05, dtype=tf.float32)
                return slim.conv2d(x, hidden_num, [kernel_size, kernel_size], stride, 'SAME',
                                   activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            else:
                return slim.conv2d(x, hidden_num, [kernel_size, kernel_size], stride, 'SAME',
                                   activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            if w_decay:
                weight_decay = tf.constant(0.05, dtype=tf.float32)
                return slim.conv2d(x, hidden_num, [kernel_size, kernel_size], stride, 'SAME',
                                   activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   biases_initializer=None,
                                   weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            else:
                return slim.conv2d(x, hidden_num, [kernel_size, kernel_size], stride, 'SAME',
                                   activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   biases_initializer=None)


def dense(x, hidden_num):
    return tf.layers.dense(x, hidden_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())


def prelu(x, name='PReLU'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5

    return pos + neg


def lrelu(x, alpha=0.3, name='LeakyReLU'):
    with tf.variable_scope(name):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def batch_norm(x, is_train=True):
    return slim.batch_norm(x, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                           scale=False, fused=True, is_training=is_train)


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
