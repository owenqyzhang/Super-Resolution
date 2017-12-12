from models import *
from utils import *


def SRResNet(inputs, targets, flags):
    # parameter container
    network = collections.namedtuple('network',
                                     'content_loss, gen_grads_and_vars, gen_output, train, global_step, learning_rate')

    # generator
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator_SRResNet(inputs, output_channel, reuse=False, flags=flags)
        gen_output.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

    # loss
    if flags.perceptual_mode == 'VGG54':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=False, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=True, scope=scope)
    elif flags.perceptual_mode == 'VGG22':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=False, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=True, scope=scope)
    elif flags.perceptual_mode == 'MSE':
        ext_feat_gen = gen_output
        ext_feat_tar = targets
    else:
        raise NotImplementedError('Unknown loss type')

    with tf.variable_scope('generator_loss'):
        with tf.variable_scope('content_loss'):
            # L2 distance of features
            diff = ext_feat_gen - ext_feat_tar
            if flags.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = flags.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

            gen_loss = content_loss

    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate)
        inc_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_opt = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
            gen_grads_and_vars = gen_opt.compute_gradients(gen_loss, gen_vars)
            gen_train = gen_opt.apply_gradients(gen_grads_and_vars)

    exp_avg = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_avg.apply([content_loss])

    return network(content_loss=exp_avg.average(content_loss),
                   gen_grads_and_vars=gen_grads_and_vars,
                   gen_output=gen_output,
                   train=tf.group(update_loss, inc_global_step, gen_train),
                   global_step=global_step,
                   learning_rate=learning_rate)


def EDSR(inputs, targets, flags):
    # parameter container
    network = collections.namedtuple('network',
                                     'content_loss, gen_grads_and_vars, gen_output, train, global_step, learning_rate')

    # generator
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator_EDSR(inputs, output_channel, reuse=False, flags=flags)
        gen_output.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

    # loss
    if flags.perceptual_mode == 'VGG54':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=False, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=True, scope=scope)
    elif flags.perceptual_mode == 'VGG22':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=False, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=True, scope=scope)
    elif flags.perceptual_mode == 'MSE':
        ext_feat_gen = gen_output
        ext_feat_tar = targets
    else:
        raise NotImplementedError('Unknown loss type')

    with tf.variable_scope('generator_loss'):
        with tf.variable_scope('content_loss'):
            # L2 distance of features
            diff = ext_feat_gen - ext_feat_tar
            if flags.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = flags.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

            gen_loss = content_loss

    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate)
        inc_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_opt = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
            gen_grads_and_vars = gen_opt.compute_gradients(gen_loss, gen_vars)
            gen_train = gen_opt.apply_gradients(gen_grads_and_vars)

    exp_avg = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_avg.apply([content_loss])

    return network(content_loss=exp_avg.average(content_loss),
                   gen_grads_and_vars=gen_grads_and_vars,
                   gen_output=gen_output,
                   train=tf.group(update_loss, inc_global_step, gen_train),
                   global_step=global_step,
                   learning_rate=learning_rate)
