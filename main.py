from networks import *

import time

Flags = tf.app.flags

# System parameters
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint.')
Flags.DEFINE_string('summary_dir', None, 'The directory to store the summaries.')
Flags.DEFINE_string('mode', 'train', 'Mode for running: train, test, or inference.')
Flags.DEFINE_string('checkpoint_SRResNet', None, 'If provided, the weights will be'
                                                 'restored from the provided checkpoint.')
Flags.DEFINE_string('checkpoint_EDSR', None, 'If provided, the weights will be restored'
                                             'from the provided checkpoint.')
Flags.DEFINE_string('checkpoint_ensemble', None, 'If provided,'
                                                 'the weights will be restored from the provided checkpoint.')
Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global steps will'
                                                 'still start from 0. If set False, global steps will be restored as'
                                                 'well.')
Flags.DEFINE_string('pre_trained_model_type', None, 'The type of pretrained model (SRResNet, EDSR, ensemble,'
                                                    'SRGAN, or EDSRGAN).')
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing, Inference => False')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'Path to vgg19 checkpoint.')
Flags.DEFINE_string('task', 'EDSR', 'SRResNet, EDSR, ensemble, SRGAN or EDSRGAN')

# Data preprocessing configurations
Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch.')
Flags.DEFINE_string('input_dir_LR', 'data/RAISE_LR', 'The directory of the low resolution input data.')
Flags.DEFINE_string('input_dir_HR', 'data/RAISE_HR', 'The directory of the high resolution input data.')
Flags.DEFINE_boolean('flip', True, 'Random flip data augmentation.')
Flags.DEFINE_boolean('random_crop', True, 'Random crop data augmentation.')
Flags.DEFINE_integer('crop_size', 24, 'Crop size of the training image.')
Flags.DEFINE_integer('name_queue_capacity', 4096, 'The capacity of the filename queue.')
Flags.DEFINE_integer('image_queue_capacity', 4096, 'The capacity of the image queue.')
Flags.DEFINE_integer('queue_thread', 8, 'The threads of the queue.')

# Generator configurations
Flags.DEFINE_integer('num_resblocks', 16, 'Number of residual blocks in the generator.')

# Content loss configurations
Flags.DEFINE_string('perceptual_mode', 'MSE', 'The type of feature used in perceptual loss.')
Flags.DEFINE_float('EPS', 1e-12, 'A small number added to prevent zero division.')
Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss.')
Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor of perceptual loss.')

# Training parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training.')
Flags.DEFINE_integer('decay_step', 500000, 'Steps needed to decay the learning rate.')
Flags.DEFINE_float('decay_rate', 0.1, 'Decay rate of learning rate.')
Flags.DEFINE_boolean('stair', False, 'Whether to perform staircase decay.')
Flags.DEFINE_float('beta', 0.9, 'Beta parameter for adam optimizer.')
Flags.DEFINE_integer('max_epoch', None, 'Max number of training.')
Flags.DEFINE_integer('max_iter', 1000000, 'Max iterations of training.')
Flags.DEFINE_integer('display_freq', 20, 'Display frequency of training.')
Flags.DEFINE_integer('summary_freq', 100, 'Frequency of writing summary.')
Flags.DEFINE_integer('save_freq', 10000, 'Frequency of saving images.')

FLAGS = Flags.FLAGS

print_configuration_op(FLAGS)

if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed.')

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

if FLAGS.mode == 'test':
    if FLAGS.task in ['SRGAN', 'SRResNet']:
        if FLAGS.checkpoint_SRResNet is None:
            raise ValueError('the checkpoint not provided.')
    if FLAGS.task in ['EDSRGAN', 'EDSR']:
        if FLAGS.checkpoint_EDSR is None:
            raise ValueError('the checkpoint not provided.')
    if FLAGS.task == 'ensemble':
        if FLAGS.checkpoint_SRResNet is None or FLAGS.checkpoint_EDSR is None or FLAGS.checkpoint_ensemble is None:
            raise ValueError('the checkpoint not provided.')

    if FLAGS.flip:
        FLAGS.flip = False

    if FLAGS.crop_size != 0:
        FLAGS.crop_size = 0

    test_data = test_data_loader(FLAGS)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    targets_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='targets_raw')
    path_lr = tf.placeholder(tf.string, shape=[], name='path_LR')
    path_hr = tf.placeholder(tf.string, shape=[], name='path_HR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResNet':
            gen_output = generator_SRResNet(inputs_raw, 3, reuse=False, flags=FLAGS)
        elif FLAGS.task == 'EDSR' or FLAGS.task == 'EDSRGAN':
            gen_output = generator_EDSR(inputs_raw, 3, reuse=False, flags=FLAGS)
        elif FLAGS.task == 'ensemble':
            gen_output_SRGAN = generator_SRResNet(inputs_raw, 3, reuse=False, flags=FLAGS)
            gen_output_EDSR = generator_EDSR(inputs_raw, 3, reuse=False, flags=FLAGS)
            gen_output = ensemble_net(gen_output_SRGAN, gen_output_EDSR, flags=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finished building the network.')

    with tf.variable_scope('convert_image'):
        inputs = deprocess_lr(inputs_raw)
        targets = deprocess(targets_raw)
        outputs = deprocess(gen_output)

        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.variable_scope('compute_psnr'):
        psnr = compute_psnr(converted_targets, converted_outputs)

    with tf.variable_scope('encode_image'):
        save_fetch = {
            'path_LR': path_lr,
            'path_HR': path_hr,
            'inputs': tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            'outputs': tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs'),
            'targets': tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name='target_pngs'),
            'PSNR': psnr
        }

    if FLAGS.task in ['SRGAN', 'SRResNet']:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        weight_initializer = tf.train.Saver(var_list)
    if FLAGS.task in ['EDSRGAN', 'EDSR']:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        weight_initializer = tf.train.Saver(var_list)
    if FLAGS.task == 'ensemble':
        SRResNet_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/SRResNet_generator_unit')
        SRResNet_weight_initializer = tf.train.Saver(SRResNet_var_list)

        EDSR_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/EDSR_generator_unit')
        EDSR_weight_initializer = tf.train.Saver(EDSR_var_list)

        ensemble_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/ensemble_net')
        ensemble_weight_initializer = tf.train.Saver(ensemble_var_list)

    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print('Loading weights from pre-trained model')
        if FLAGS.task in ['SRResNet', 'SRGAN']:
            weight_initializer.restore(sess, FLAGS.checkpoint_SRResNet)
        elif FLAGS.task in ['EDSR', 'EDSRGAN']:
            weight_initializer.restore(sess, FLAGS.checkpoint_EDSR)
        else:
            SRResNet_weight_initializer.restore(sess, FLAGS.checkpoint_SRResNet)
            EDSR_weight_initializer.restore(sess, FLAGS.checkpoint_EDSR)
            ensemble_weight_initializer.restore(sess, FLAGS.checkpoin_ensemble)

        max_iter = len(test_data.inputs)
        print('Evaluation starts')
        for i in range(max_iter):
            input_im = np.array([test_data.inputs[i]]).astype(np.float32)
            target_im = np.array([test_data.targets[i]]).astype(np.float32)
            path_lr_test = test_data.paths_LR[i]
            path_hr_test = test_data.paths_HR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, targets_raw: target_im,
                                                      path_lr: path_lr_test, path_hr: path_hr_test})
            filesets = save_image(results, FLAGS)
            for j, f in enumerate(filesets):
                print('evaluate image', f['name'], 'PSNR: ', results['PSNR'])

elif FLAGS.mode == 'inference':
    if FLAGS.task in ['SRGAN', 'SRResNet']:
        if FLAGS.checkpoint_SRResNet is None:
            raise ValueError('the checkpoint not provided.')
    if FLAGS.task in ['EDSRGAN', 'EDSR']:
        if FLAGS.checkpoint_EDSR is None:
            raise ValueError('the checkpoint not provided.')
    if FLAGS.task == 'ensemble':
        if FLAGS.checkpoint_SRResNet is None or FLAGS.checkpoint_EDSR is None or FLAGS.checkpoint_ensemble is None:
            raise ValueError('the checkpoint not provided.')

    if FLAGS.flip:
        FLAGS.flip = False

    if FLAGS.crop_size != 0:
        FLAGS.crop_size = 0

    test_data = test_data_loader(FLAGS)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_lr = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResNet':
            gen_output = generator_SRResNet(inputs_raw, 3, reuse=False, flags=FLAGS)
        elif FLAGS.task == 'EDSR' or FLAGS.task == 'EDSRGAN':
            gen_output = generator_EDSR(inputs_raw, 3, reuse=False, flags=FLAGS)
        elif FLAGS.task == 'ensemble':
            gen_output_SRGAN = generator_SRResNet(inputs_raw, 3, reuse=False, flags=FLAGS)
            gen_output_EDSR = generator_EDSR(inputs_raw, 3, reuse=False, flags=FLAGS)
            gen_output = ensemble_net(gen_output_SRGAN, gen_output_EDSR, flags=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finished building the network.')

    with tf.variable_scope('convert_image'):
        inputs = deprocess_lr(inputs_raw)
        outputs = deprocess(gen_output)

        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.variable_scope('encode_image'):
        save_fetch = {
            'path_LR': path_lr,
            'inputs': tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            'outputs': tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }

    if FLAGS.task in ['SRGAN', 'SRResNet']:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        weight_initializer = tf.train.Saver(var_list)
    if FLAGS.task in ['EDSRGAN', 'EDSR']:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        weight_initializer = tf.train.Saver(var_list)
    if FLAGS.task == 'ensemble':
        SRResNet_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/SRResNet_generator_unit')
        SRResNet_weight_initializer = tf.train.Saver(SRResNet_var_list)

        EDSR_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/EDSR_generator_unit')
        EDSR_weight_initializer = tf.train.Saver(EDSR_var_list)

        ensemble_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/ensemble_net')
        ensemble_weight_initializer = tf.train.Saver(ensemble_var_list)

    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print('Loading weights from pre-trained model')
        if FLAGS.task in ['SRResNet', 'SRGAN']:
            weight_initializer.restore(sess, FLAGS.checkpoint_SRResNet)
        elif FLAGS.task in ['EDSR', 'EDSRGAN']:
            weight_initializer.restore(sess, FLAGS.checkpoint_EDSR)
        else:
            SRResNet_weight_initializer.restore(sess, FLAGS.checkpoint_SRResNet)
            EDSR_weight_initializer.restore(sess, FLAGS.checkpoint_EDSR)
            ensemble_weight_initializer.restore(sess, FLAGS.checkpoin_ensemble)

        max_iter = len(test_data.inputs)
        print('Evaluation starts')
        for i in range(max_iter):
            input_im = np.array([test_data.inputs[i]]).astype(np.float32)
            path_lr_test = test_data.paths_LR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_lr: path_lr_test})
            filesets = save_image(results, FLAGS)
            for j, f in enumerate(filesets):
                print('evaluate image', f['name'])

elif FLAGS.mode == 'train':
    data = data_loader(FLAGS)
    print('Data count = %d' % data.image_count)

    if FLAGS.task == 'SRResNet':
        net = SRResNet(data.inputs, data.targets, FLAGS)
    elif FLAGS.task == 'EDSR':
        net = EDSR(data.inputs, data.targets, FLAGS)
    elif FLAGS.task == 'SRGAN':
        net = SRGAN(data.inputs, data.targets, FLAGS)
    elif FLAGS.task == 'EDSRGAN':
        net = EDSRGAN(data.inputs, data.targets, FLAGS)
    elif FLAGS.task == 'ensemble':
        if FLAGS.checkpoint_SRResNet is None or FLAGS.checkpoint_EDSR is None:
            raise ValueError('the checkpoint not provided.')

        net = ensemble(data.inputs, data.targets, FLAGS)
    else:
        raise NotImplementedError('Unknown task type')

    print('Finished building the network')

    with tf.variable_scope('convert_image'):
        inputs = deprocess_lr(data.inputs)
        targets = deprocess(data.targets)
        outputs = deprocess(net.gen_output)

        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.variable_scope('compute_psnr'):
        psnr = compute_psnr(converted_targets, converted_outputs)

    with tf.variable_scope('inputs_summary'):
        tf.summary.image('input_summary', converted_inputs)

    with tf.variable_scope('targets_summary'):
        tf.summary.image('target_summary', converted_targets)

    with tf.variable_scope('outputs_summary'):
        tf.summary.image('outputs_summary', converted_outputs)

    if FLAGS.task == 'SRGAN' or FLAGS.task == 'EDSRGAN':
        tf.summary.scalar('discriminator_loss', net.disc_loss)
        tf.summary.scalar('adversarial_loss', net.adv_loss)
        tf.summary.scalar('content_loss', net.cont_loss)
        tf.summary.scalar('generator_loss', net.cont_loss + FLAGS.ratio * net.adv_loss)
        tf.summary.scalar('PSNR', psnr)
        tf.summary.scalar('learning_rate', net.learning_rate)
    elif FLAGS.task == 'SRResNet' or FLAGS.task == 'EDSR' or FLAGS.task == 'ensemble':
        tf.summary.scalar('content_loss', net.content_loss)
        tf.summary.scalar('PSNR', psnr)
        tf.summary.scalar('learning_rate', net.learning_rate)

    saver = tf.train.Saver(max_to_keep=1)

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    if FLAGS.task == 'SRGAN':
        if FLAGS.pre_trained_model_type == 'SRGAN':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        elif FLAGS.pre_trained_model_type == 'SRResNet':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        else:
            raise ValueError('Unknown pre trained model type')
    elif FLAGS.task == 'EDSRGAN':
        if FLAGS.pre_trained_model_type == 'EDSRGAN':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        elif FLAGS.pre_trained_model_type == 'EDSR':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        else:
            raise ValueError('Unknown pre trained model type')
    elif FLAGS.task == 'SRResNet':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    elif FLAGS.task == 'EDSR':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    elif FLAGS.task == 'ensemble':
        SRResNet_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope='generator/SRResNet_generator_unit')
        SRResNet_weight_initializer = tf.train.Saver(SRResNet_var_list)

        EDSR_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator/EDSR_generator_unit')
        EDSR_weight_initializer = tf.train.Saver(EDSR_var_list)

        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ensemble')

    weight_initializer = tf.train.Saver(var_list2)

    if not FLAGS.perceptual_mode == 'MSE':
        vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
        vgg_restore = tf.train.Saver(vgg_var_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        if FLAGS.task in ['SRResNet', 'SRGAN']:
            if FLAGS.checkpoint_SRResNet is not None and FLAGS.pre_trained_model is False:
                print('Loading model from checkpoint...')
                # checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_SRResNet)
                saver.restore(sess, FLAGS.checkpoint_SRResNet)
            elif FLAGS.checkpoint_SRResNet is not None and FLAGS.pre_trained_model is True:
                print('Loading weights from pre trained model')
                weight_initializer.restore(sess, FLAGS.checkpoint_SRResNet)
        elif FLAGS.task in ['EDSR', 'EDSRGAN']:
            if FLAGS.checkpoint_EDSR is not None and FLAGS.pre_trained_model is False:
                print('Loading model from checkpoint...')
                # checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_EDSR)
                saver.restore(sess, FLAGS.checkpoint_EDSR)
            elif FLAGS.checkpoint_EDSR is not None and FLAGS.pre_trained_model is True:
                print('Loading weights from pre trained model')
                weight_initializer.restore(sess, FLAGS.checkpoint_EDSR)
        else:
            SRResNet_weight_initializer.restore(sess, FLAGS.checkpoint_SRResNet)
            EDSR_weight_initializer.restore(sess, FLAGS.checkpoint_EDSR)

            if FLAGS.checkpoint_ensemble is not None and FLAGS.pre_trained_model is False:
                print('Loading model from checkpoint...')
                checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_ensemble)
                saver.restore(sess, checkpoint)
            elif FLAGS.checkpoint_ensemble is not None and FLAGS.pre_trained_model is True:
                print('Loading weights from pre trained model')
                weight_initializer.restore(sess, FLAGS.checkpoint_ensemble)

        if not FLAGS.perceptual_mode == 'MSE':
            vgg_restore.restore(sess, FLAGS.vgg_ckpt)
            print('VGG19 restored successfully')

        if FLAGS.max_epoch is None:
            if FLAGS.max_iter is None:
                raise ValueError('One of max_epoch or max_iter should be provided.')
            else:
                max_iter = FLAGS.max_iter
        else:
            max_iter = FLAGS.max_epoch * data.steps_per_epoch

        print('Optimization starts')
        start = time.time()
        for step in range(1, max_iter + 1):
            fetches = {'train': net.train, 'global_step': sv.global_step}

            if (step % FLAGS.display_freq) == 0 and step > 1:
                if FLAGS.task == 'SRGAN' or FLAGS.task == 'EDSRGAN':
                    fetches['disc_loss'] = net.disc_loss
                    fetches['adv_loss'] = net.adv_loss
                    fetches['content_loss'] = net.cont_loss
                    fetches['PSNR'] = psnr
                    fetches['learning_rate'] = net.learning_rate
                    fetches['global_step'] = net.global_step
                elif FLAGS.task == 'SRResNet' or FLAGS.task == 'EDSR' or FLAGS.task == 'ensemble':
                    fetches['content_loss'] = net.content_loss
                    fetches['PSNR'] = psnr
                    fetches['learning_rate'] = net.learning_rate
                    fetches['global_step'] = net.global_step

            if (step % FLAGS.summary_freq) == 0 and step > 1:
                fetches['summary'] = sv.summary_op

            results = sess.run(fetches)
            step_temp = step
            step = results['global_step']

            if (step % FLAGS.summary_freq) == 0 and step_temp > 1:
                print('Recording summary')
                sv.summary_writer.add_summary(results['summary'], results['global_step'])

            if (step % FLAGS.display_freq) == 0 and step_temp > 1:
                train_epoch = math.ceil(results['global_step'] / data.steps_per_epoch)
                train_step = (results['global_step'] - 1) % data.steps_per_epoch + 1
                rate = step * FLAGS.batch_size / (time.time() - start)
                remaining = (max_iter - step) * FLAGS.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                if FLAGS.task == 'SRGAN' or FLAGS.task == 'EDSRGAN':
                    print("global_step", results["global_step"])
                    print("PSNR", results["PSNR"])
                    print("discrim_loss", results["disc_loss"])
                    print("adversarial_loss", results["adv_loss"])
                    print("content_loss", results["content_loss"])
                    print("learning_rate", results['learning_rate'])
                elif FLAGS.task == 'SRResNet' or FLAGS.task == 'EDSR' or FLAGS.task == 'ensemble':
                    print("global_step", results["global_step"])
                    print("PSNR", results["PSNR"])
                    print("content_loss", results["content_loss"])
                    print("learning_rate", results['learning_rate'])

            if (step % FLAGS.save_freq) == 0 and step_temp > 1:
                print('Save the checkpoint')
                saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=sv.global_step)

            if step >= max_iter:
                break

        print('Optimization done!')
