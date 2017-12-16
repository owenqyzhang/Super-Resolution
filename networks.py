from models import *
from utils import *


def SRResNet(inputs, targets, flags):
    # parameter container
    network = collections.namedtuple('network',
                                     'content_loss, gen_grads_and_vars, gen_output, train, global_step, learning_rate')

    # generator
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator_SRResNet(inputs, output_channel, reuse=tf.AUTO_REUSE, flags=flags)
        gen_output.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

    # loss
    if flags.perceptual_mode == 'VGG54':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
    elif flags.perceptual_mode == 'VGG22':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
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
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
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
        gen_output = generator_EDSR(inputs, output_channel, reuse=tf.AUTO_REUSE, flags=flags)
        gen_output.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

    # loss
    if flags.perceptual_mode == 'VGG54':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
    elif flags.perceptual_mode == 'VGG22':
        with tf.variable_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
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
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(diff), axis=[3]))
            else:
                content_loss = flags.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

            gen_loss = content_loss

    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
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


def SRGAN(inputs, targets, flags):
    network = collections.namedtuple('network', 'disc_real, disc_fake, disc_loss, disc_grads_vars, adv_loss, cont_loss,\
                                                gen_grads_vars, gen_output, train, global_step, learning_rate')

    # generator
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator_SRResNet(inputs, output_channel, reuse=tf.AUTO_REUSE, flags=flags)
        gen_output.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            disc_fake_output = discriminator(gen_output, flags=flags)

    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            disc_real_output = discriminator(targets, flags=flags)

    # loss
    if flags.perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
    elif flags.perceptual_mode == 'VGG22':
        with tf.name_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
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

        with tf.variable_scope('adversarial_loss'):
            adversarial_loss = tf.reduce_mean(-tf.log(disc_fake_output + flags.EPS))

        gen_loss = content_loss + flags.ratio * adversarial_loss

    with tf.variable_scope('discriminator_loss'):
        disc_fake_loss = tf.log(1 - disc_fake_output + flags.EPS)
        disc_real_loss = tf.log(disc_real_output + flags.EPS)

        disc_loss = tf.reduce_mean(-(disc_fake_loss + disc_real_loss))

    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
        inc_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('discriminator_train'):
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        disc_opt = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
        disc_grads_and_vars = disc_opt.compute_gradients(disc_loss, disc_vars)
        disc_train = disc_opt.apply_gradients(disc_grads_and_vars)

    with tf.variable_scope('generator_train'):
        with tf.control_dependencies([disc_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_opt = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
            gen_grads_and_vars = gen_opt.compute_gradients(gen_loss, gen_vars)
            gen_train = gen_opt.apply_gradients(gen_grads_and_vars)

    exp_avg = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_avg.apply([disc_loss, adversarial_loss, content_loss])

    return network(disc_real=disc_real_output,
                   disc_fake=disc_fake_output,
                   disc_loss=exp_avg.average(disc_loss),
                   disc_grads_vars=disc_grads_and_vars,
                   adv_loss=exp_avg.average(adversarial_loss),
                   cont_loss=exp_avg.average(content_loss),
                   gen_grads_vars=gen_grads_and_vars,
                   gen_output=gen_output,
                   train=tf.group(update_loss, inc_global_step, gen_train),
                   global_step=global_step,
                   learning_rate=learning_rate)


def EDSRGAN(inputs, targets, flags):
    network = collections.namedtuple('network', 'disc_real, disc_fake, disc_loss, disc_grads_vars, adv_loss, cont_loss,\
                                                gen_grads_vars, gen_output, train, global_step, learning_rate')

    # generator
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator_EDSR(inputs, output_channel, reuse=tf.AUTO_REUSE, flags=flags)
        gen_output.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            disc_fake_output = discriminator(gen_output, flags=flags)

    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            disc_real_output = discriminator(targets, flags=flags)

    # loss
    if flags.perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
    elif flags.perceptual_mode == 'VGG22':
        with tf.name_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
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

        with tf.variable_scope('adversarial_loss'):
            adversarial_loss = tf.reduce_mean(-tf.log(disc_fake_output + flags.EPS))

        gen_loss = content_loss + flags.ratio * adversarial_loss

    with tf.variable_scope('discriminator_loss'):
        disc_fake_loss = tf.log(1 - disc_fake_output + flags.EPS)
        disc_real_loss = tf.log(disc_real_output + flags.EPS)

        disc_loss = tf.reduce_mean(-(disc_fake_loss + disc_real_loss))

    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
        inc_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('discriminator_train'):
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        disc_opt = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
        disc_grads_and_vars = disc_opt.compute_gradients(disc_loss, disc_vars)
        disc_train = disc_opt.apply_gradients(disc_grads_and_vars)

    with tf.variable_scope('generator_train'):
        with tf.control_dependencies([disc_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_opt = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
            gen_grads_and_vars = gen_opt.compute_gradients(gen_loss, gen_vars)
            gen_train = gen_opt.apply_gradients(gen_grads_and_vars)

    exp_avg = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_avg.apply([disc_loss, adversarial_loss, content_loss])

    return network(disc_real=disc_real_output,
                   disc_fake=disc_fake_output,
                   disc_loss=exp_avg.average(disc_loss),
                   disc_grads_vars=disc_grads_and_vars,
                   adv_loss=exp_avg.average(adversarial_loss),
                   cont_loss=exp_avg.average(content_loss),
                   gen_grads_vars=gen_grads_and_vars,
                   gen_output=gen_output,
                   train=tf.group(update_loss, inc_global_step, gen_train),
                   global_step=global_step,
                   learning_rate=learning_rate)


def ensemble(inputs, targets, flags):
    # parameter container
    network = collections.namedtuple('network',
                                     'content_loss, gen_grads_and_vars, gen_output, train, global_step, learning_rate')

    # generator
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output_SRGAN = generator_SRResNet(inputs, output_channel, reuse=tf.AUTO_REUSE, flags=flags)
        gen_output_SRGAN.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

        gen_output_EDSR = generator_EDSR(inputs, output_channel, reuse=tf.AUTO_REUSE, flags=flags)
        gen_output_EDSR.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

    with tf.variable_scope('ensemble'):
        gen_output = ensemble_net(gen_output_SRGAN, gen_output_EDSR, flags=flags)

    # loss
    if flags.perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
    elif flags.perceptual_mode == 'VGG22':
        with tf.name_scope('vgg19_1') as scope:
            ext_feat_gen = vgg19_slim(gen_output, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            ext_feat_tar = vgg19_slim(targets, flags.perceptual_mode, reuse=tf.AUTO_REUSE, scope=scope)
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
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
        inc_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ensemble')
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
