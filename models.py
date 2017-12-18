from ops import *


def generator_SRResNet(gen_inputs, gen_output_channels, reuse=False, flags=None):
    if flags is None:
        raise ValueError('No FLAGS is provided for generator')

    def res_block(x, output_channel, stride, scope):
        with tf.variable_scope(scope):
            x0 = conv(x, output_channel, 3, stride, use_bias=False, scope='conv_1')
            x0 = batch_norm(x0, flags.is_training)
            x0 = prelu(x0)
            x0 = conv(x0, output_channel, 3, stride, use_bias=False, scope='conv_2')
            x0 = batch_norm(x0, flags.is_training)
            x = tf.add(x, x0)

        return x

    with tf.variable_scope('SRResNet_generator_unit', reuse=reuse):
        with tf.variable_scope('input_stage'):
            x1 = conv(gen_inputs, hidden_num=64, kernel_size=9, stride=1)
            x1 = prelu(x1)

        stage1 = x1

        for i in range(1, flags.num_resblocks + 1, 1):
            name_scope = 'resblock' + str(i)
            x1 = res_block(x1, output_channel=64, stride=1, scope=name_scope)

        with tf.variable_scope('resblock_output'):
            x1 = conv(x1, 64, 3, 1)
            x1 = batch_norm(x1, flags.is_training)

        x1 = tf.add(x1, stage1)

        with tf.variable_scope('sub_pixel_conv_stage1'):
            x1 = conv(x1, hidden_num=256)
            x1 = pixel_shuffler(x1, scale=2)
            x1 = prelu(x1)

        with tf.variable_scope('sub_pixel_conv_stage2'):
            x1 = conv(x1, hidden_num=256)
            x1 = pixel_shuffler(x1, scale=2)
            x1 = prelu(x1)

        with tf.variable_scope('output_stage'):
            x1 = conv(x1, gen_output_channels, kernel_size=9)

    return x1


def generator_EDSR(gen_inputs, gen_output_channels, reuse=False, flags=None):
    if flags is None:
        raise ValueError('No FLAGS is provided for generator')

    def res_block(x, output_channel, stride, scale, scope):
        with tf.variable_scope(scope):
            x0 = conv(x, output_channel, 3, stride, use_bias=False, scope='conv_1')
            x0 = tf.nn.relu(x0)
            x0 = conv(x0, output_channel, 3, stride, use_bias=False, scope='conv_2')
            x0 = tf.multiply(x0, tf.constant(scale, dtype=tf.float32))
            x = tf.add(x, x0)

        return x

    with tf.variable_scope('EDSR_generator_unit', reuse=reuse):
        with tf.variable_scope('normalization'):
            mean_x = tf.reduce_mean(gen_inputs)
            gen_inputs = tf.subtract(gen_inputs, mean_x)

        with tf.variable_scope('input_stage'):
            x1 = conv(gen_inputs, hidden_num=256, kernel_size=3, stride=1)

        stage1 = x1

        scaling_factor = 0.1
        for i in range(1, flags.num_resblocks + 1, 1):
            name_scope = 'resblock' + str(i)
            x1 = res_block(x1, scale=scaling_factor, output_channel=256, stride=1, scope=name_scope)

        with tf.variable_scope('resblock_output'):
            x1 = conv(x1, hidden_num=256, kernel_size=3, stride=1)

        x1 = tf.add(x1, stage1)

        with tf.variable_scope('sub_pixel_conv_stage1'):
            x1 = conv(x1, hidden_num=256)
            x1 = pixel_shuffler(x1, scale=2)

        with tf.variable_scope('sub_pixel_conv_stage2'):
            x1 = conv(x1, hidden_num=256)
            x1 = pixel_shuffler(x1, scale=2)

        with tf.variable_scope('output_stage'):
            x1 = conv(x1, gen_output_channels, kernel_size=9)

        with tf.variable_scope('denormalization'):
            x1 = tf.add(x1, mean_x)

    return x1


def discriminator(dis_inputs, flags=None):
    if flags is None:
        raise ValueError('No FLAGS is provided for generator')

    def discriminator_block(x, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            x1 = conv(x, kernel_size=kernel_size, hidden_num=output_channel, stride=stride)
            x1 = batch_norm(x1, flags.is_training)
            x1 = lrelu(x1, 0.2)

        return x1

    with tf.variable_scope('discriminator'):
        with tf.variable_scope('input_stage'):
            x1 = conv(dis_inputs)
            x1 = lrelu(x1, alpha=0.2)

        x1 = discriminator_block(x1, 64, 3, 2, 'disblock_1')

        x1 = discriminator_block(x1, 128, 3, 1, 'disblock_2')

        x1 = discriminator_block(x1, 128, 3, 2, 'disblock_3')

        x1 = discriminator_block(x1, 256, 3, 1, 'disblock_4')

        x1 = discriminator_block(x1, 256, 3, 2, 'disblock_5')

        x1 = discriminator_block(x1, 512, 3, 1, 'disblock_6')

        x1 = discriminator_block(x1, 512, 3, 2, 'disblock_7')

        with tf.variable_scope('dense_layer_1'):
            x1 = slim.flatten(x1)
            x1 = dense(x1, 1024)
            x1 = lrelu(x1, alpha=0.2)

        with tf.variable_scope('dense_layer_2'):
            x1 = dense(x1, 1)
            x1 = tf.nn.sigmoid(x1)

    return x1


def ensemble_net(input_1, input_2, flags=None):
    if flags is None:
        raise ValueError('No FLAGS is provided for generator')

    with tf.variable_scope('ensemble_net'):
        with tf.variable_scope('input_stage'):
            x = tf.concat([input_1, input_2], axis=3)

        with tf.variable_scope('conv1'):
            x = conv(x, hidden_num=8, kernel_size=3, stride=1)
            x = tf.nn.relu(x)

        with tf.variable_scope('conv2'):
            x = conv(x, hidden_num=16, kernel_size=3, stride=1)
            x = tf.nn.relu(x)

        with tf.variable_scope('conv3'):
            x = conv(x, hidden_num=32, kernel_size=3, stride=1)
            x = tf.nn.relu(x)

        with tf.variable_scope('conv4'):
            x = conv(x, hidden_num=3, kernel_size=3, stride=1)

    return x


def vgg19_slim(inputs, loss_type, reuse, scope):
    if loss_type == 'VGG54':
        # target_layer = scope.name + '/vgg_19/conv5/conv5_4'
        target_layer = 'vgg_19/conv5/conv5_4'
    elif loss_type == 'VGG22':
        # target_layer = scope.name + '/vgg_19/conv2/conv2_2'
        target_layer = 'vgg_19/conv2/conv2_2'
    else:
        NotImplementedError('Unknown perceptual loss type')
    _, output = vgg_19(inputs, reuse=reuse)
    output = output[target_layer]

    return output
