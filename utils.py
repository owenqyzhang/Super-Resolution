import tensorflow as tf
import collections
import os
import math


def preprocess(image):
    with tf.variable_scope('preprocess'):
        return image * 2 - 1


def deprocess(image):
    with tf.variable_scope('deprocess'):
        return (image + 1) / 2


def preprocess_lr(image):
    with tf.variable_scope('preprocess_lr'):
        return tf.identity(image)


def deprocess_lr(image):
    with tf.variable_scope('deprocess_lr'):
        return tf.identity(image)


def random_flip(image, decision):
    f1 = tf.identity(image)
    f2 = tf.image.flip_left_right(image)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)

    return output


def data_loader(flags):
    with tf.device('/cpu:0'):
        # data batch
        data = collections.namedtuple('Data',
                                      'paths_LR, paths_HR, inputs, targets, image_count, steps_per_epoch')

        # input directory check
        if flags.input_dir_LR == 'None' or flags.input_dir_HR == 'None':
            raise ValueError('Input directory is not provided.')

        if not os.path.exists(flags.input_dir_LR) or not os.path.exists(flags.input_dir_HR):
            raise ValueError('Input directory not found.')

        image_list_lr = os.listdir(flags.input_dir_LR)
        image_list_lr = [_ for _ in image_list_lr if _.endswith('.png')]
        if len(image_list_lr) == 0:
            raise Exception('No png files in the input directory.')

        image_list_lr_temp = sorted(image_list_lr)
        image_list_lr = [os.path.join(flags.input_dir_LR, _) for _ in image_list_lr_temp]
        image_list_hr = [os.path.join(flags.input_dir_HR, _) for _ in image_list_lr_temp]

        image_list_lr_tensor = tf.convert_to_tensor(image_list_lr, dtype=tf.string)
        image_list_hr_tensor = tf.convert_to_tensor(image_list_hr, dtype=tf.string)

        with tf.variable_scope('load_image'):
            # input image list queue
            output = tf.train.slice_input_producer([image_list_lr_tensor, image_list_hr_tensor],
                                                   shuffle=False, capacity=flags.name_queue_capacity)

            # reading and decoding the images
            reader = tf.WholeFileReader(name='image_reader')
            image_lr = tf.read_file(output[0])
            image_hr = tf.read_file(output[1])
            input_image_lr = tf.image.decode_png(image_lr, channels=3)
            input_image_hr = tf.image.decode_png(image_hr, channels=3)
            input_image_lr = tf.image.convert_image_dtype(input_image_lr, dtype=tf.float32)
            input_image_hr = tf.image.convert_image_dtype(input_image_hr, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(input_image_lr)[2], 3, message='image does not have 3 channels')
            with tf.control_dependencies([assertion]):
                input_image_lr = tf.identity(input_image_lr)
                input_image_hr = tf.identity(input_image_hr)

            # normalize the low resolution image to [0, 1], high resolution to [-1, 1]
            a_image = preprocess_lr(input_image_lr)
            b_image = preprocess(input_image_hr)

            inputs, targets = [a_image, b_image]

        # data augmentation
        with tf.variable_scope('data_preprocessing'):
            with tf.variable_scope('random_crop'):
                if flags.random_crop and flags.mode == 'train':
                    print('[Config] Use random crop')
                    input_size = tf.shape(inputs)
                    # target_size = tf.shape(targets)
                    offset_w = tf.cast(tf.floor(tf.random_uniform([], 0,
                                                                  tf.cast(input_size[1],
                                                                          tf.float32) - flags.crop_size)),
                                       dtype=tf.int32)
                    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0,
                                                                  tf.cast(input_size[0],
                                                                          tf.float32) - flags.crop_size)),
                                       dtype=tf.int32)

                    inputs = tf.image.crop_to_bounding_box(inputs, offset_h, offset_w,
                                                           flags.crop_size, flags.crop_size)
                    targets = tf.image.crop_to_bounding_box(targets, offset_h, offset_w,
                                                            flags.crop_size, flags.crop_size)
                else:
                    inputs = tf.identity(inputs)
                    targets = tf.identity(targets)

            with tf.variable_scope('random_flip'):
                if flags.flip and flags.mode == 'train':
                    print('[Config] Use random flip')
                    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)

                    input_images = random_flip(inputs, decision)
                    target_images = random_flip(targets, decision)
                else:
                    input_images = tf.identity(inputs)
                    target_images = tf.identity(targets)

            input_images.set_shape([flags.crop_size, flags.crop_size, 3])
            input_images.set_shape([flags.crop_size * 4, flags.crop_size * 4, 3])

            if flags.mode == 'train':
                paths_lr_batch, paths_hr_batch, inputs_batch, targets_batch = tf.train.shuffle_batch(
                    [output[0], output[1], input_images, target_images], batch_size=flags.batch_size,
                    capacity=flags.image_queue_capacity + 4 * flags.batch_size,
                    min_after_dequeue=flags.image_queue_capacity, num_threads=flags.queue_thread)
            else:
                paths_lr_batch, paths_hr_batch, inputs_batch, targets_batch = tf.train.batch(
                    [output[0], output[1], input_images, target_images],
                    batch_size=flags.batch_size, num_threads=flags.queue_thread, allow_smaller_final_batch=True)

            steps_per_epoch = int(math.ceil(len(image_list_lr) / flags.batch_size))
            inputs_batch.set_shape([flags.batch_size, flags.crop_size, flags.crop_size, 3])
            target_images.set_shape([flags.batch_size, flags.crop_size * 4, flags.crop_size * 4, 3])

        return data(paths_LR=paths_lr_batch,
                    paths_HR=paths_hr_batch,
                    inputs=inputs_batch,
                    targets=targets_batch,
                    image_count=len(image_list_lr),
                    steps_per_epoch=steps_per_epoch)


def save_image(fetches, flags, step=None):
    image_dir = os.path.join(flags.output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = fetches['path_LR']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {'name': name, 'step': step}

    if flags.mode == 'inference':
        kind = 'outputs'
        filename = name + '.png'
        if step is not None:
            filename = '%08d-%s' % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, 'wb') as f:
            f.write(contents)
        filesets.append(fileset)
    else:
        for kind in ['inputs', 'outputs', 'targets']:
            filename = name + '_' + kind + '.png'
            if step is not None:
                filename = '%08d-%s' % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][0]
            with open(out_path, 'wb') as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets
