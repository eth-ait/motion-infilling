import ast

import numpy as np
import os
import tensorflow as tf
from tbase import utils
from tbase.data_loader import Databases
from tbase.data_loader import LoaderV2
from tbase.data_loader import FeederV2
from tbase.skeleton import Skeleton

import flags_parser
from models import Conv1dAutoencoder
from models import VGGAutoencoder
from models import VanillaAutoencoder
from models import InceptionAutoencoder
from models import HoldenAutoencoder

fs = tf.app.flags
flags_parser.define_model_name(fs, 'default_model')
flags_parser.define_normalizer(fs, 'tanh')
flags_parser.define_perturbator(fs, 'none')
flags_parser.define_perturbation_amount(fs, 15)
flags_parser.define_perturbation_size(fs, '[2, 4, 6]')
flags_parser.define_seed(fs, 42)
flags_parser.define_if_test(fs)
flags_parser.define_if_masked_loss(fs)
flags_parser.define_batch_size(fs, 10)
flags_parser.define_model(fs)
flags_parser.define_discard_foot_contacts(fs)

fs.DEFINE_integer('n_epochs', 25, 'number of epochs to train [25]')
fs.DEFINE_float('lr', 0.001, 'learning rate for Adam Optimizer [0.001]')
fs.DEFINE_integer('decay_lr_every', 0, 'If bigger than 0, then the learning rate is decayed every so steps [0]')
fs.DEFINE_float('dropout', 0.8, 'keep probability for dropout regularization [0.8]')
fs.DEFINE_float('dropout_internal', 1.0, ' if smaller than 1.0, then add a dropout layer before each convolution/deconvolution [1.0]')
fs.DEFINE_string('activation', 'tanh', 'which activation function to choose for the hidden layers [tanh].')
fs.DEFINE_string('out_activation', 'tanh', 'which activation to use for the last layer in the decoder or none [tanh]')
fs.DEFINE_string('loss_fn', 'L2', 'which loss function to use, "L1", "L2" or "cross-entropy" [L2]')
fs.DEFINE_float('eta', 0.0, 'scalar weight for bone length constraint, disabled if set to 0.0 [0.0]')
fs.DEFINE_float('alpha', 0.0, 'scalar weight for smoothness loss, disabled if set to 0.0 [0.0]')
fs.DEFINE_string('feature_maps', '[32, 64, 128, 256, 512]', 'sizes of the feature maps per layer ["[32, 64, 128, 256, 512]"]')
fs.DEFINE_bool('use_batch_norm', False, 'if set, uses batch normalization for every convolutional layer')
fs.DEFINE_string('description', 'default', 'an optional description/remark that will be saved into a file in the checkpoint directory')
fs.DEFINE_string('curriculum', 'none', 'if set, defines the curriculum to be used with ColumnPerturbatorGaussian consisting'
                                       'of 6 comma-separated integers: [start_mean_width, end_mean_width, mean_std, '
                                       'increase_every, increase_amount, start_epoch], [none]')
fs.DEFINE_bool('no_weight_sharing', False, 'if set, weights between decoder and encoder are NOT shared')
fs.DEFINE_bool('use_channels', False, 'if set, joints in the input are organized in channels')
fs.DEFINE_integer('store_to_run', -1, 'if set to >= 0 the model will be saved to checkpoint_dir/model_name/save_to_run '
                                      'instead of the next available run id. [-1]')
fs.DEFINE_bool('use_same_filter_size', False, 'if set, all layers have same filter size [False]')
fs.DEFINE_string('filter_heights', '3,3,3,3,3', 'height of the filter, per layer ["3,3,3,3,3"]')
fs.DEFINE_string('filter_widths', '3,3,3,3,3', 'width of the filter, per layer ["3,3,3,3,3"]')
fs.DEFINE_bool('use_unpooling', False, 'if set, the decoder uses a custom unpooling operation instead of strided '
                                       'deconvolutions (only works with --model vanilla).')
fs.DEFINE_string('initializer', 'normal', 'initializer for weights in convolutional layers, "normal" or "xavier" [normal]')
fs.DEFINE_integer('skip_connections', 0, 'if > 0, the decoder will use this amount of skip connections, currently only implemented for the VGG model')
fs.DEFINE_boolean('add_joint_perturbator', False, 'if set the curriculum will also perturbate up to 3 joints')


FLAGS = fs.FLAGS
RNG = np.random.RandomState(FLAGS.seed)


def main(argv):
    if len(argv) > 1:
        # we have some unparsed flags
        raise ValueError('unknown flags: {}'.format(' '.join(argv[1:])))

    # Get the normalizer
    normalizer = flags_parser.get_normalizer(FLAGS)()

    # Load the data depending on which preprocessed data we are using
    data_path = flags_parser.get_data_path()
    loader = LoaderV2(train_path=os.path.join(data_path, 'train'),
                      valid_path=os.path.join(data_path, 'valid'),
                      normalizer=normalizer,
                      discard_foot_contacts=FLAGS.discard_foot_contacts)
    data_train, data_valid = loader.get_data_unnormalized(Databases.EDIN_LOCOMOTION) if FLAGS.test else loader.load_all()
    feeder = FeederV2(data_train=data_train,
                      data_valid=data_valid,
                      batch_size=FLAGS.batch_size)
    data = data_train  # hack, is ok because in subsequent alls only used to get sequence length and pose dim

    # get the curriculum and associated perturbator
    curriculum, perturbator = flags_parser.get_curriculum(FLAGS)

    # if curriculum does not specify the perturbator, get the one specified in the flags
    perturbator = flags_parser.get_perturbator(FLAGS) if curriculum is None else perturbator

    # create the model
    model_type = FLAGS.model.lower()
    if model_type == 'vanilla':
        autoencoder = VanillaAutoencoder
    elif model_type == 'conv1d':
        autoencoder = Conv1dAutoencoder
    elif model_type == 'vgg':
        autoencoder = VGGAutoencoder
    elif model_type == 'inception':
        autoencoder = InceptionAutoencoder
    elif model_type == 'holden':
        autoencoder = HoldenAutoencoder
    else:
        raise ValueError('model type "{}" unknown'.format(FLAGS.model))

    rest_dim = data.shape[1] - len(Skeleton.ALL_JOINTS)*3
    pose_dim = len(Skeleton.ALL_JOINTS) + rest_dim if FLAGS.use_channels else data.shape[1]
    seq_length = data.shape[2]
    channels = 3 if FLAGS.use_channels else 1

    store_to_run = FLAGS.store_to_run if FLAGS.store_to_run >= 0 else None

    filter_widths = ast.literal_eval(FLAGS.filter_widths)
    if len(filter_widths) == 1:
        num_layers = len(ast.literal_eval(FLAGS.feature_maps))
        filter_widths *= num_layers
    filter_heights = ast.literal_eval(FLAGS.filter_heights)
    if len(filter_heights) == 1:
        num_layers = len(ast.literal_eval(FLAGS.feature_maps))
        filter_heights *= num_layers
        
    autoencoder = autoencoder.build_from_scratch(height=pose_dim,
                                                 width=seq_length,
                                                 channels=channels,
                                                 feature_map_sizes=ast.literal_eval(FLAGS.feature_maps),
                                                 activation=FLAGS.activation,
                                                 out_activation=FLAGS.out_activation,
                                                 eta=FLAGS.eta,
                                                 alpha=FLAGS.alpha,
                                                 loss_fn=FLAGS.loss_fn,
                                                 optimizer='adam',
                                                 checkpoint_dir=flags_parser.get_checkpoints_path(),
                                                 name=FLAGS.name,
                                                 perturbator=perturbator,
                                                 use_masked_loss=FLAGS.use_masked_loss,
                                                 use_batch_norm=FLAGS.use_batch_norm,
                                                 share_weights=not FLAGS.no_weight_sharing,
                                                 store_to_run=store_to_run,
                                                 filter_widths=filter_widths,
                                                 filter_heights=filter_heights,
                                                 lr=FLAGS.lr,
                                                 lr_decay=FLAGS.decay_lr_every,
                                                 use_unpooling=FLAGS.use_unpooling,
                                                 initializer=FLAGS.initializer,
                                                 skip_connections=FLAGS.skip_connections)

    # save the computed normalization into the model checkpoint
    normalizer.save(autoencoder.save_path)

    # count number of trainable parameters
    trainable_params = utils.count_trainable_parameters()
    print('{} trainable parameters'.format(trainable_params))
    print('saving checkpoints to {}'.format(autoencoder.save_path))

    # create a description string that will be saved into the checkpoint dir
    desc = utils.to_printable_string(**FLAGS.__flags, optimizer='adam', trainable_params=trainable_params)

    # train
    autoencoder.train(data_feeder=feeder,
                      n_epochs=FLAGS.n_epochs,
                      dropout=FLAGS.dropout,
                      dropout_internal=FLAGS.dropout_internal,
                      curriculum=curriculum,
                      description=desc)


if __name__ == '__main__':
    tf.app.run()
