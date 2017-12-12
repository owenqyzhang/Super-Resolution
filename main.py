from ops import *
from utils import *

Flags = tf.app.flags

# System parameters
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint.')
Flags.DEFINE_string('summary_dir', None, 'The directory to store the summaries.')
Flags.DEFINE_string('mode', 'train', 'Mode for running: train, test, or inference.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weights will be restored from the provided checkpoint.')
Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global steps will'
                                                 'still start from 0. If set False, global steps will be restored as'
                                                 'well.')
Flags.DEFINE_string('pre_trained_model_type', 'SRResNet', 'The type of pretrained model (SRResNet, EDSR, ensemble,'
                                                          'SRGAN, or EDSRGAN).')
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing, Inference => False')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'Path to vgg19 checkpoint.')
Flags.DEFINE_string('task', None, 'SRResNet, EDSR, ensemble, SRGAN or EDSRGAN')

# Data preprocessing configurations
Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch.')
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the low resolution input data.')
Flags.DEFINE_string('intpu_dir_HR', None, 'The directory of the high resolution input data.')
Flags.DEFINE_boolean('flip', True, 'Random flip data augmentation.')
Flags.DEFINE_boolean('random_crop', True, 'Random crop data augmentation.')
Flags.DEFINE_integer('crop_size', 24, 'Crop size of the training image.')
Flags.DEFINE_integer('name_queue_capacity', 4096, 'The capacity of the filename queue.')
Flags.DEFINE_integer('image_queue_capacity', 4096, 'The capacity of the image queue.')
Flags.DEFINE_integer('queue_thread', 32, 'The threads of the queue.')

# Generator configurations
Flags.DEFINE_integer('num_resblock', 16, 'Number of residual blocks in the generator.')

# Content loss configurations
Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss.')
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

if FLAGS.mode == 'train':
    data = data_loader(FLAGS)
    print('Data count = %d' % data.image_count)
