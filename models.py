from ops import *


def generator_SRResNet(gen_inputs, gen_output_channels, reuse=False, flags=None):
    if flags is None:
        raise ValueError('No FLAGS is provided for generator')

    with tf.variable_scope('SRResNet_generator_unit', reuse=reuse):
        with tf.variable_scope('input_stage'):
            x1 = conv(gen_inputs, hidden_num=64, kernel_size=9, stride=1)
            x1 = prelu(x1)

        stage1 = x1

        for i in range(1, flags.num_resblocks + 1, 1):
            name_scope = 'resblock' + str(i)
            x1 = res_block(x1, output_channel=64, stride=1, scope=name_scope, flags=flags)

        with tf.variable_scope('resblock_output'):
            x1 = conv(x1)
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

    with tf.variable_scope('EDSR_generator_unit', reuse=reuse):
        with tf.variable_scope('input_stage'):
            x1 = conv(gen_inputs, hidden_num=256, kernel_size=3, stride=1)
            # x1 = prelu(x1)

        stage1 = x1

        scaling_factor = 0.1
        for i in range(1, flags.num_resblocks + 1, 1):
            name_scope = 'resblock' + str(i)
            x1 = res_block_edsr(x1, scale=scaling_factor, output_channel=256, stride=1, scope=name_scope)

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

    return x1


def discriminator(dis_inputs, flags=None):
    if flags is None:
        raise ValueError('No FLAGS is provided for generator')

    with tf.variable_scope('discriminator'):
        with tf.variable_scope('input_stage'):
            x1 = conv(dis_inputs)
            x1 = lrelu(x1, alpha=0.2)

        x1 = discriminator_block(x1, 64, 3, 2, 'disblock_1', flags)

        x1 = discriminator_block(x1, 128, 3, 1, 'disblock_2', flags)

        x1 = discriminator_block(x1, 128, 3, 2, 'disblock_3', flags)

        x1 = discriminator_block(x1, 256, 3, 1, 'disblock_4', flags)

        x1 = discriminator_block(x1, 256, 3, 2, 'disblock_5', flags)

        x1 = discriminator_block(x1, 512, 3, 1, 'disblock_6', flags)

        x1 = discriminator_block(x1, 512, 3, 2, 'disblock_7', flags)

        with tf.variable_scope('dense_layer_1'):
            x1 = tf.reshape(x1, [x1.get_shape().as_list()[0], -1])
            x1 = dense(x1, 1024, flags.is_training)
            x1 = lrelu(x1, alpha=0.2)

        with tf.variable_scope('dense_layer_2'):
            x1 = dense(x1, 1, flags.is_training)
            x1 = tf.nn.sigmoid(x1)

    return x1


def vgg19_slim(inputs, loss_type, reuse, scope):
    if loss_type == 'VGG54':
        target_layer = scope + 'vgg_19/conv5/conv5_4'
    elif loss_type == 'VGG22':
        target_layer = scope + 'vgg_19/conv2/conv2_2'
    else:
        NotImplementedError('Unknown perceptual loss type')
    _, output = vgg_19(inputs, reuse=reuse)
    output = output[target_layer]

    return output
