import datetime
import os
import re
import time

import numpy as np
import tensorflow as tf
from tbase import utils
from tbase.skeleton import Skeleton
from tbase.data_loader import to_3_channels_rep
from tbase.data_loader import to_1_channel_rep

from ops import batch_normalization
from ops import conv1d_strided
from ops import conv2d_strided
from ops import activation_fn_from_string
from ops import deconv1d_strided
from ops import deconv2d_strided
from ops import unpool
from plotter import plot_reconstruction


def _get_next_available_save_path_id(base_path):
    max_counter = -1
    root, dirs, files = next(os.walk(base_path), (None, None, None))

    if dirs is None:
        return 0

    for sub_name in dirs:
        if 'run_' in sub_name:
            counter = re.findall('^.*_([0-9]+)$', sub_name)[0]
            max_counter = max(int(counter), max_counter)
    return max_counter + 1


def _shift_one_frame_tf(tensor):
    """
    Shifts the input tensor in the third dimension one entry to the right. First entry is retained, last one is dropped.
    :param tensor: A tensor of rank 3.
    :return: The shifted tensor of rank 3.
    """
    assert len(tensor.get_shape()) == 3
    t_slice = tf.slice(tensor, [0, 0, 0], [-1, -1, tensor.get_shape()[2].value - 1])
    t_first = tf.slice(tensor, [0, 0, 0], [-1, -1, 1])
    t_shifted = tf.concat(2, [t_first, t_slice])
    return t_shifted


def _compute_bone_lengths_tf(data):
    """
    Same as `compute_bone_lengths_np` but implemented in tensorflow.
    """
    # resize `data` to (batch_size, n_joints, 3, sequence_length)
    parents = Skeleton.PARENTS
    n_joints = len(parents)
    n_batches = tf.shape(data)[0]
    joints_only = tf.slice(data, [0, 0, 0], [-1, n_joints * 3, -1])
    v = tf.reshape(joints_only, shape=[n_batches, n_joints, 3, data.get_shape()[-1].value])

    # get start and end indices for each bone
    j1, j2 = list(zip(*Skeleton.BONES))

    def _gather_axis(params, indices, axis):
        return tf.stack(tf.unstack(tf.gather(tf.unstack(params, axis=axis), indices=indices), axis=0), axis=axis)

    # in the second dimension, select those joints with the indices given by j1 and j2.
    t_j1 = _gather_axis(v, j1, axis=1)
    t_j2 = _gather_axis(v, j2, axis=1)

    # compute bone length
    actual_bones = tf.subtract(t_j1, t_j2)
    actual_lengths = tf.sqrt(tf.reduce_sum(tf.multiply(actual_bones, actual_bones), axis=2))

    return actual_lengths


def compute_bone_lengths_np(data):
    """
    Computes the length of each bone for each batch and frame in the input data.
    :param data: A np array of size (n_batches, dim, seq_length).
    :return A np array of size (n_batches, n_bones, seq_length) giving the length of each bone for all batches and
      frames.
    """
    # bones is list of (start_idx, end_idx)
    j1, j2 = list(zip(*Skeleton.BONES))

    # reshape input data to (n_batches, n_joints, 3, seq_length)
    n_batches = data.shape[0]
    n_joints = len(Skeleton.ALL_JOINTS)
    seq_length = data.shape[2]
    r = np.reshape(data[:, :n_joints * 3], [n_batches, n_joints, 3, seq_length])

    # get start and end positions
    start_jts = r[:, j1]
    end_jts = r[:, j2]

    # compute the length of the bones
    actual_bones = start_jts - end_jts
    actual_lengths = np.sqrt(np.sum(np.multiply(actual_bones, actual_bones), axis=2))

    return actual_lengths


def shift_one_frame(data):
    """
    Shift input data one frame to the right, first frame is just retained, last one is dropped.
    :param data: A np array of size (n_batches, dim, seq_length)
    :return: A np array of the same size as `data` but shifted one frame to the right (i.e. in the third dimension)
    """
    shifted = np.copy(data)
    shifted[:, :, 1:] = shifted[:, :, :-1]
    return shifted


class SavableModel(object):
    """
    Base class for models that can be saved and restored. Essentially just contains a few shared helper functions.
    """

    def __init__(self, checkpoint_dir, name, restored=False, store_to_run=False):
        """
        Constructor.

        :param checkpoint_dir: Path to root of checkpoints directory. Checkpoints will be stored in checkpoint_dir/name
           and tensorboard logs will be stored in checkpoint_dir/name/logs.
        :param name: Name of the model.
        :param restored: If True, means that this a restored model that will be used to continue training or for tests.
        :param store_to_run: If set, the run directory is determined by this integer. 
        """
        # base path is checkpoint_dir/model_name
        self.base_path = os.path.join(checkpoint_dir, name)
        self.name = name
        self.restored = restored
        self.saver = None  # subclasses responsibility to initialize saver

        self._create_output_dirs(store_to_run)

    def _create_output_dirs(self, store_to_run=None):
        if store_to_run is not None:
            x = store_to_run
        else:
            # construct a save path of the form checkpoint_dir/model_name/run_x where x is a unique identifier
            x = _get_next_available_save_path_id(self.base_path)

            if self.restored:
                # we don't want to create a new run-subdirectory, but continue with the latest available
                x -= 1

        # directory to save checkpoints
        self.save_path = os.path.join(self.base_path, 'run_{:0>3}'.format(x))

        if os.path.exists(self.save_path) and not self.restored:
            # check that the save_path directory does not yet exist, otherwise we would be overwriting
            raise ValueError('save directory "{}" already exists.'.format(self.save_path))

        # directory to save tensorboard logs
        self.log_path = os.path.join(self.base_path, 'logs_{:0>3}'.format(x))

        # directory to store visualizations
        self.visual_path = os.path.join(self.save_path, 'visuals')

    def prepare_training(self):
        # create required directories if they don't exist yet
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.visual_path):
            os.makedirs(self.visual_path)

    def dump_configuration(self, *args):
        """
        Save parameters of interest to a file in `self.save_path`.
        """
        file_name = os.path.join(self.save_path, 'config.txt')
        with open(file_name, 'w') as f:
            for a in args:
                f.write('{}\n'.format(a))

            f.write('hg revision: {}\n'.format(utils.get_current_hg_revision()))
            now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
            f.write('mka, {}'.format(now))

    def append_to_configuration(self, *args):
        """
        Appends strings to the configuration file of this model.
        """
        file_name = os.path.join(self.save_path, 'config.txt')
        with open(file_name, 'a') as f:
            f.write('\nAddendum\n')
            for a in args:
                f.write('{}\n'.format(a))

    def save(self, sess, global_step):
        save_path = os.path.join(self.save_path, self.name)
        self.saver.save(sess, save_path=save_path, global_step=global_step)

    def load(self, sess, checkpoint):
        self.saver.restore(sess, checkpoint)
        self.restored = True


class VanillaAutoencoder(SavableModel):
    """
    Build a trainable vanilla autoencoder network to encode images.
    """
    # define some custom graph keys to store tensors of interest in collections
    gk_inputs = 'VAE_inputs'
    gk_results = 'VAE_results'
    gk_loss_op = 'VAE_loss_op'
    gk_train_op = 'VAE_train_op'
    gk_valid = 'VAE_valid'
    gk_summaries = 'VAE_summaries'
    gk_global_step = 'VAE_global_step'

    def __init__(self, checkpoint_dir, name, perturbator, use_masked_loss, restored=False, store_to_run=None):
        """
        Constructor.

        :param checkpoint_dir: Path to root of checkpoints directory. Checkpoints will be stored in checkpoint_dir/name
           and tensorboard logs will be stored in checkpoint_dir/name/logs.
        :param name: Name of the model.
        :param perturbator: A `Perturbator` instance that perturbates data or None if no perturbation should be used.
        :param use_masked_loss: If True, loss will only be calculated on the corrupted input area.
        :param restored: If True, means that this a restored model that will be used to continue training or for tests.
        :param store_to_run: If set, the run directory is determined by this integer.
        """
        super().__init__(checkpoint_dir, name, restored, store_to_run)
        self.perturbator = perturbator
        self.use_masked_loss = use_masked_loss

        # the following members must all be defined if the instance is to be used for training or inference
        self.input_ = None  # input to the encoder
        self.dropout = None  # keep probability to use during training
        self.dropout_internal = None  # keep probability before internal conv/deconv operations to use during training
        self.is_training = None  # set if this is a training run, needed for batch normalization
        self.targets = None  # placeholder for the targets to be reconstructed
        self.bone_targets = None  # placeholder for the bone length targets
        self.smoothness_margin = None  # placeholder for margin used in smoothness loss
        self.mask = None  # placeholder for the corruption mask used for loss calculation
        self.latent_code = None  # latent representation of the input
        self.reconstructed = None  # the output of the decoder
        self.perturbation_width_pl = None  # width of the column perturbation used to normalize the loss
        self.update_perturbation_width = None  # operation to update the current perturbation width
        self.loss_op = None  # the operation computing the loss
        self.l2_loss = None  # the l2 reconstruction loss (not used in the optimization)
        self.train_op = None  # the operation performing the optimization
        self.v_loss_pl = None  # placeholder to feed validation loss computed externally into the model
        self.update_v_loss = None  # operation allowing the set the externally computed validation loss
        self.v_l2_loss_pl = None  # placeholder to feed L2 validation loss computed externally into the model
        self.update_v_l2_loss = None  # operation allowing to set the externally computed L2 validation loss
        self.train_summaries = None  # tensorboard summaries used for training stats
        self.valid_summaries = None  # tensorboard summaries used for validation stats
        self.global_step = None  # variable holding the global step
        self.update_global_step = None  # operation used to increment global step by one
        self.saver = None  # saver object to write checkpoints

    @classmethod
    def build_from_scratch(cls, height, width, channels, feature_map_sizes, activation, out_activation, eta, alpha,
                           loss_fn, optimizer, checkpoint_dir, name, perturbator, use_masked_loss, use_batch_norm,
                           share_weights, store_to_run, filter_widths, filter_heights, lr, lr_decay, use_unpooling,
                           initializer, skip_connections):
        """
        Builds the whole computational graph from scratch. For parameter description cf. `build_model(..)`.

        :param checkpoint_dir: Path to root of checkpoints directory. Checkpoints will be stored in checkpoint_dir/name
           and tensorboard logs will be stored in checkpoint_dir/name/logs.
        :param name: Name of the model.
        :param perturbator: A `Perturbator` instance that perturbates data or None if no perturbation should be used.
        :param use_masked_loss: If True, loss will only be calculated on the corrupted input area.
        :param use_batch_norm: If True, adds a batch-normalization layer to all convolutional layers.
        :param share_weights: If True, learnable parameters are shared between encoder and decoder.
        :param filter_widths: List of width of the convolutional filter, per layer.
        :param filter_heights: List of height of the convolutional filter, per layer.
        :return: The created instance.
        """
        obj = cls(checkpoint_dir, name, perturbator, use_masked_loss, restored=False, store_to_run=store_to_run)
        obj.build_model(height, width, channels, feature_map_sizes, activation, out_activation, use_batch_norm,
                        eta, alpha, loss_fn, optimizer, share_weights, filter_widths, filter_heights, lr, lr_decay,
                        use_unpooling, initializer, skip_connections)
        return obj

    @classmethod
    def build_from_metagraph(cls, checkpoint, perturbator, use_masked_loss):
        """
        Instead of building from scratch, the graph is imported from the meta file called `checkpoint` + '.meta'. This
        is useful if the model is to be used for inference or when training should be resumed. Note that this
        function only builds the graph, i.e. it does NOT restore the value of the variables. To do so, call `load()`.

        :param checkpoint: Path to a directory where the '.meta' file is stored, which is used to import the graph.
          The path is expected in the following format: /some/path/checkpoints/model_name/run_XYZ/model_name. I.e.
          the `.meta` file lies directly in the folder run_XYZ. This is e.g. the output of `tf.train.latest_checkpoint`.
        :param perturbator: A `Perturbator` instance that perturbates data or None if no perturbation should be used.
        :param use_masked_loss: If True, loss will only be calculated on the corrupted input area.
        :return: The created instance.
        """
        run_dir = os.path.dirname(checkpoint)
        model_name_dir = os.path.dirname(run_dir)
        checkpoint_dir = os.path.dirname(model_name_dir)
        name = os.path.basename(model_name_dir)
        obj = cls(checkpoint_dir, name, perturbator, use_masked_loss, restored=True)
        obj.saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)

        def get_collection_entry(coll_name, idx, default=None):
            """
            Checks if the requested collection entry is available and returns `default` otherwise. This is
            necesseray to be backwards-compatible.
            """
            coll = tf.get_collection_ref(coll_name)
            if len(coll) >= idx + 1:
                return coll[idx]
            else:
                return default

        obj.input_ = get_collection_entry(VanillaAutoencoder.gk_inputs, 0)
        obj.dropout = get_collection_entry(VanillaAutoencoder.gk_inputs, 1)
        obj.targets = get_collection_entry(VanillaAutoencoder.gk_inputs, 2, obj.input_)
        obj.mask = get_collection_entry(VanillaAutoencoder.gk_inputs, 3,
                                        default=tf.placeholder(tf.float32,
                                                               shape=obj.targets.get_shape(),
                                                               name='mask_pl'))
        obj.is_training = get_collection_entry(VanillaAutoencoder.gk_inputs, 4,
                                               default=tf.placeholder(tf.bool,
                                                                      shape=[],
                                                                      name='is_training_pl'))
        obj.bone_targets = get_collection_entry(VanillaAutoencoder.gk_inputs, 5,
                                                default=tf.placeholder(tf.float32,
                                                                       shape=[None,
                                                                              Skeleton.N_BONES,
                                                                              obj.targets.get_shape()[-1].value],
                                                                       name='bone_targets_pl'))
        obj.smoothness_margin = get_collection_entry(VanillaAutoencoder.gk_inputs, 6,
                                                     default=tf.placeholder(tf.float32,
                                                                            shape=[None],
                                                                            name='smoothness_margin_pl'))
        obj.dropout_internal = get_collection_entry(VanillaAutoencoder.gk_inputs, 7, default=tf.placeholder(tf.float32,
                                                                                                            shape=[],
                                                                                                            name='dropout_internal_pl'))
        obj.latent_code = get_collection_entry(VanillaAutoencoder.gk_results, 0)
        obj.reconstructed = get_collection_entry(VanillaAutoencoder.gk_results, 1)
        obj.loss_op = get_collection_entry(VanillaAutoencoder.gk_loss_op, 0)
        obj.l2_loss = get_collection_entry(VanillaAutoencoder.gk_loss_op, 1,
                                           default=tf.constant(0.0))
        obj.perturbation_width_pl = get_collection_entry(VanillaAutoencoder.gk_loss_op, 2,
                                                         tf.placeholder(tf.float32, shape=[], name='pert_width_pl'))
        obj.update_perturbation_width = get_collection_entry(VanillaAutoencoder.gk_loss_op, 3,
                                                             default=tf.constant(0.0))
        obj.train_op = get_collection_entry(VanillaAutoencoder.gk_train_op, 0)
        obj.v_loss_pl = get_collection_entry(VanillaAutoencoder.gk_valid, 0)
        obj.update_v_loss = get_collection_entry(VanillaAutoencoder.gk_valid, 1)
        obj.v_l2_loss_pl = get_collection_entry(VanillaAutoencoder.gk_valid, 2,
                                                default=tf.placeholder(tf.float32, shape=[], name='v_l2_loss_pl'))
        obj.update_v_l2_loss = get_collection_entry(VanillaAutoencoder.gk_valid, 3,
                                                    default=tf.constant(0.0))
        obj.train_summaries = get_collection_entry(VanillaAutoencoder.gk_summaries, 0)
        obj.valid_summaries = get_collection_entry(VanillaAutoencoder.gk_summaries, 1)
        obj.global_step = get_collection_entry(VanillaAutoencoder.gk_global_step, 0)
        obj.update_global_step = get_collection_entry(VanillaAutoencoder.gk_global_step, 1)

        return obj

    def build_model(self, height, width, channels, feature_map_sizes, activation, out_activation,
                    use_batch_norm, eta, alpha, loss_fn, optimizer, share_weights, filter_widths, filter_heights,
                    lr, lr_decay, use_unpooling, initializer, skip_connections):
        """
        Builds the computational graph.

        :param height: Height of the input image.
        :param width: Width of the input image.
        :param channels: Number of channels of the input image.
        :param feature_map_sizes: List of number of output channels to be used for every layer.
        :param activation: String specifying the activation function, `sigmoid`, `tanh` or `relu`.
        :param out_activation: String specifying which activation function to use for the output layer or None
        :param use_batch_norm: If True, adds a batch-normalization layer to all convolutional layers.
        :param eta: Scalar weight >= 0.0 for bone length loss. If 0.0 bone length loss is disabled. 
        :param alpha: Scalar weight >= 0.0 for smoothness loss. If 0.0 smoothness loss is disabled. 
        :param loss_fn: Which loss function to use, can currently be either 'L1', 'L2' or 'cross-entropy'.
        :param optimizer: The Tensorflow optimizer to use during training.
        :param share_weights: If True, learnable parameters are shared between encoder and decoder.
        :param filter_widths: List of width of the convolutional filter, per layer.
        :param filter_heights: List of height of the convolutional filter, per layer.
        :param use_unpooling: If True, decoder will use custom unpooling operation instead of strided deconv.
        :param initializer: String specifying which initializer to use to initialize convolutional weights.
        :param skip_connections: Amount of skip connections to be used by decoder.
        """
        activation_fn = activation_fn_from_string(activation)
        out_activation_fn = activation_fn_from_string(out_activation)

        with tf.name_scope('input'):
            # placeholders
            self.input_ = tf.placeholder(tf.float32, shape=[None, height, None, channels], name='input_pl')
            self.dropout = tf.placeholder(tf.float32, shape=[], name='dropout_pl')
            self.dropout_internal = tf.placeholder(tf.float32, shape=[], name='dropout_internal_pl')
            self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training_pl')

            # use dropout only on input
            input_d = tf.nn.dropout(self.input_, keep_prob=self.dropout)

        # build encoder
        recn, recn_before, e_layers, d_layers = self.build_autoencoder(input_=input_d,
                                                                       encoder_feature_sizes=feature_map_sizes,
                                                                       filter_widths=filter_widths,
                                                                       filter_heights=filter_heights,
                                                                       activation_fn=activation_fn,
                                                                       out_activation=out_activation_fn,
                                                                       share_weights=share_weights,
                                                                       use_batch_norm=use_batch_norm,
                                                                       is_training=self.is_training,
                                                                       dropout_internal=self.dropout_internal,
                                                                       use_unpooling=use_unpooling,
                                                                       initializer=initializer,
                                                                       skip_connections=skip_connections)
        self.reconstructed = recn
        self.latent_code = e_layers[-1]

        # keep track of global step inside graph
        with tf.name_scope('global_step'):
            self.global_step = tf.Variable(tf.constant(0), trainable=False, name='global_step')
            self.update_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1),
                                                name='update_global_step')

        self.add_optimizer(optimizer=optimizer,
                           loss_fn=loss_fn,
                           eta=eta,
                           alpha=alpha,
                           reconstructed_before_activation=recn_before,
                           lr=lr,
                           lr_decay=lr_decay)

        # store tensors of interest in collections
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.input_)
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.dropout)
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.targets)
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.mask)
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.is_training)
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.bone_targets)
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.smoothness_margin)
        tf.add_to_collection(VanillaAutoencoder.gk_inputs, self.dropout_internal)
        tf.add_to_collection(VanillaAutoencoder.gk_results, self.latent_code)
        tf.add_to_collection(VanillaAutoencoder.gk_results, self.reconstructed)
        tf.add_to_collection(VanillaAutoencoder.gk_global_step, self.global_step)
        tf.add_to_collection(VanillaAutoencoder.gk_global_step, self.update_global_step)

    def add_optimizer(self, optimizer, loss_fn, eta, alpha, lr, lr_decay, reconstructed_before_activation=None):
        """
        Add optimizer to the model.
        :param eta: Scalar weight >= 0.0 for bone length loss. If 0.0 bone length loss is disabled. 
        :param alpha: Scalar weight >= 0.0 for smoothness loss. If 0.0 smoothness loss is disabled. 
        :param loss_fn: Which loss function to use, can currently be either 'L1', 'L2' or 'cross-entropy'.
        :param optimizer: String saying which otpimizer to use.
        :param lr: Learning rate for the optimizer.
        :param lr_decay: Learning rate decay or 0 if no decay should be used.
        :param reconstructed_before_activation: Dirty hack to make cross-entropy work, should be a member variable
        """

        def _l1(targets, reconstructed, mask):
            diff = tf.multiply(tf.subtract(targets, reconstructed), mask)
            loss_per_batch = tf.reduce_sum(tf.abs(diff), axis=[1, 2, 3])
            return loss_per_batch

        def _l2(targets, reconstructed, mask):
            diff = tf.multiply(tf.subtract(targets, reconstructed), mask)
            loss_per_batch = tf.reduce_sum(tf.multiply(diff, diff), axis=[1, 2, 3])
            return loss_per_batch

        def _get_loss_per_batch(loss_fn, targets, reconstructed, mask):
            fn = loss_fn.lower()
            if fn == 'l1':
                return _l1(targets, reconstructed, mask)
            elif fn == 'l2':
                return _l2(targets, reconstructed, mask)
            elif fn == 'cross-entropy':
                ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(reconstructed_before_activation, targets)
                loss_per_batch = tf.reduce_mean(tf.multiply(ce_loss, mask), axis=[1, 2, 3])
            else:
                raise ValueError('loss function "{}" unknown'.format(loss_fn))
            return loss_per_batch

        def _get_bone_length_loss_per_batch(target_lengths, reconstructed):
            reconstructed_lengths = _compute_bone_lengths_tf(reconstructed)
            diff = tf.subtract(target_lengths, reconstructed_lengths)
            loss_per_batch = tf.sqrt(tf.reduce_sum(tf.multiply(diff, diff), axis=[1, 2, 3]))
            return loss_per_batch

        def _get_smoothness_loss_per_batch(reconstructed, margin):
            # shift reconstructed loss one frame to the right
            shifted = _shift_one_frame_tf(reconstructed)
            # define smoothness loss as difference between reconstructed and shifted must be small (in L2 norm)
            diff = tf.subtract(reconstructed, shifted)
            loss_per_batch = tf.sqrt(tf.reduce_sum(tf.multiply(diff, diff), axis=[1, 2, 3]))
            # reconstruction cannot be `smoother` than the original
            loss_per_batch = tf.maximum(tf.subtract(loss_per_batch, margin), 0.0)
            return loss_per_batch

        # define the loss function
        with tf.name_scope('loss_fn'):
            # define some input placeholders
            self.targets = tf.placeholder(tf.float32,
                                          shape=self.input_.get_shape(),
                                          name='targets_pl')
            self.bone_targets = tf.placeholder(tf.float32,
                                               shape=[None, Skeleton.N_BONES, self.input_.get_shape()[2].value],
                                               name='bone_targets_pl')
            self.smoothness_margin = tf.placeholder(tf.float32,
                                                    shape=[None],
                                                    name='smoothness_margin_pl')
            self.mask = tf.placeholder(tf.float32,
                                       shape=self.targets.get_shape(),
                                       name='mask_pl')

            # compute the configured loss
            loss_per_batch = _get_loss_per_batch(loss_fn, self.targets, self.reconstructed, self.mask)
            main_loss = tf.reduce_mean(loss_per_batch)

            # add bone length constraint if configured
            if eta > 0.0:
                bone_length_loss = _get_bone_length_loss_per_batch(self.bone_targets, self.reconstructed)
                b_loss = tf.reduce_mean(bone_length_loss)
            else:
                b_loss = 0.0

            # add smoothness loss if configured
            if alpha > 0.0:
                smoothness_loss = _get_smoothness_loss_per_batch(self.reconstructed, self.smoothness_margin)
                s_loss = tf.reduce_mean(smoothness_loss)
            else:
                s_loss = 0.0

            corr_width = tf.Variable(1.0, trainable=False)
            self.perturbation_width_pl = tf.placeholder(tf.float32, shape=[], name='pert_width_pl')
            self.update_perturbation_width = tf.assign(corr_width, self.perturbation_width_pl, name='update_pert_width')
            self.loss_op = tf.divide(tf.add(tf.add(main_loss, tf.multiply(eta, b_loss)),
                                            tf.multiply(alpha, s_loss)),
                                     corr_width, name='loss_op')

            # no matter which loss we use, always compute L2 loss as well (this is not being optimized, just visualized)
            self.l2_loss = tf.reduce_mean(_l2(self.targets, self.reconstructed, self.mask))

        with tf.name_scope('optimizer'):
            if lr_decay > 0:
                learning_rate = tf.train.exponential_decay(lr, self.global_step, lr_decay,
                                                           0.96, staircase=True)
            else:
                learning_rate = lr

            if optimizer.lower() == 'adam':
                optim = tf.train.AdamOptimizer(learning_rate)
            else:
                raise ValueError('optimizer {} unknown'.format(optimizer))

            # compute gradients and extract their norm
            grads_and_vars = optim.compute_gradients(self.loss_op)
            grads, _ = list(zip(*grads_and_vars))
            norms = tf.global_norm(grads)
            gradnorm_s = tf.summary.scalar('gradient norm', norms)
            self.train_op = optim.apply_gradients(grads_and_vars, name='train_op')

        with tf.name_scope('valid_loss'):
            # create a dummy variable through which we can store the validation loss
            v_loss = tf.Variable(tf.constant(0.0), trainable=False)
            self.v_loss_pl = tf.placeholder(tf.float32, shape=[], name='v_loss_pl')
            self.update_v_loss = tf.assign(v_loss, self.v_loss_pl, name='update_v_loss')

            # also create a dummy variable for L2 loss, no matter which main loss we are using
            v_l2_loss = tf.Variable(tf.constant(0.0), trainable=False)
            self.v_l2_loss_pl = tf.placeholder(tf.float32, shape=[], name='v_l2_loss_pl')
            self.update_v_l2_loss = tf.assign(v_l2_loss, self.v_l2_loss_pl, name='update_v_l2_loss')

        with tf.name_scope('train_summaries'):
            m_loss_s = tf.summary.scalar('main_loss', main_loss)
            b_loss_s = tf.summary.scalar('bone_length_loss', b_loss)
            s_loss_s = tf.summary.scalar('smoothness_loss', s_loss)
            t_loss_s = tf.summary.scalar('total_loss', self.loss_op)
            l2_loss_s = tf.summary.scalar('loss_l2', self.l2_loss)
            width_s = tf.summary.scalar('perturbation_width', corr_width)
            self.train_summaries = tf.summary.merge([m_loss_s, gradnorm_s, b_loss_s, s_loss_s,
                                                     t_loss_s, l2_loss_s, width_s],
                                                    name='train_summaries')

        with tf.name_scope('valid_summaries'):
            v_loss_s = tf.summary.scalar('validation_loss', v_loss)
            v_l2_loss_s = tf.summary.scalar('validation_l2_loss', v_l2_loss)
            self.valid_summaries = tf.summary.merge([v_loss_s, v_l2_loss_s], name='valid_summaries')

        # must instantiate saver only when full graph is built, otherwise not all variables are stored
        self.saver = tf.train.Saver()

        tf.add_to_collection(VanillaAutoencoder.gk_loss_op, self.loss_op)
        tf.add_to_collection(VanillaAutoencoder.gk_loss_op, self.l2_loss)
        tf.add_to_collection(VanillaAutoencoder.gk_loss_op, self.perturbation_width_pl)
        tf.add_to_collection(VanillaAutoencoder.gk_loss_op, self.update_perturbation_width)
        tf.add_to_collection(VanillaAutoencoder.gk_train_op, self.train_op)
        tf.add_to_collection(VanillaAutoencoder.gk_valid, self.v_loss_pl)
        tf.add_to_collection(VanillaAutoencoder.gk_valid, self.update_v_loss)
        tf.add_to_collection(VanillaAutoencoder.gk_valid, self.v_l2_loss_pl)
        tf.add_to_collection(VanillaAutoencoder.gk_valid, self.update_v_l2_loss)
        tf.add_to_collection(VanillaAutoencoder.gk_summaries, self.train_summaries)
        tf.add_to_collection(VanillaAutoencoder.gk_summaries, self.valid_summaries)

    @classmethod
    def build_autoencoder(cls, input_, encoder_feature_sizes, filter_widths, filter_heights,
                          activation_fn, out_activation, share_weights, use_batch_norm, is_training,
                          dropout_internal, use_unpooling, initializer, skip_connections):
        """
        Builds the autoencoder and returns the reconstructed input.
        :param input_: The input to the encoder (placeholder).
        :param encoder_feature_sizes: The number of filters for each layer of the encoder.
        :param filter_widths: List of widths for the filter in each layer.
        :param filter_heights: List of heights for the filter in each layer. 
        :param activation_fn: Activation function to be used.
        :param out_activation: Activation function to be used on the last layer of the decoder.
        :param share_weights: If set, weights between decoder and encoder are shared.
        :param use_batch_norm: If set, batch normalization is applied.
        :param is_training: If set, this is a training run (required for batch normalization).
        :param dropout_internal: Keep probability to be used for dropout applied to hidden layers.
        :param use_unpooling: If True, decoder will use custom unpooling operation instead of strided deconv.
        :param initializer: String specifying which initializer to use.
        :param skip_connections: Amount of skip connections to be used by decoder.
        :return: The reconstruction and additionally the hidden layers of the encoder and decoder.
        """
        # build encoder
        with tf.name_scope('encoder'):
            e_hidden_layers = cls.build_encoder(input_=input_,
                                                activation_fn=activation_fn,
                                                feature_map_sizes=encoder_feature_sizes,
                                                use_batch_norm=use_batch_norm,
                                                filter_widths=filter_widths,
                                                filter_heights=filter_heights,
                                                is_training=is_training,
                                                dropout_internal=dropout_internal,
                                                initializer=initializer)
            e_hidden_layers = [input_] + e_hidden_layers

        # build decoder
        with tf.name_scope('decoder'):
            d_hidden_layers, last_before_activation = cls.build_decoder(e_hidden_layers=e_hidden_layers,
                                                                        activation_fn=activation_fn,
                                                                        out_activation=out_activation,
                                                                        use_batch_norm=use_batch_norm,
                                                                        share_weights=share_weights,
                                                                        filter_widths=filter_widths,
                                                                        filter_heights=filter_heights,
                                                                        is_training=is_training,
                                                                        dropout_internal=dropout_internal,
                                                                        use_unpooling=use_unpooling,
                                                                        initializer=initializer,
                                                                        skip_connections=skip_connections)

        reconstructed = d_hidden_layers[-1]
        reconstructed_before_activation = last_before_activation

        # e_hidden_layers might contain tuples, get rid of them
        hidden_layers = [v[0] if isinstance(v, tuple) else v for v in e_hidden_layers]

        return reconstructed, reconstructed_before_activation, hidden_layers, d_hidden_layers

    @staticmethod
    def build_encoder(input_, activation_fn, feature_map_sizes, use_batch_norm,
                      filter_widths, filter_heights, is_training, dropout_internal, initializer):
        """
        Builds the encoder.
        :param input_: A tensor of shape (batch_size, height, width, channels).
        :param activation_fn: Which activation function to use for each layer (directly the Tensorflow op).
        :param feature_map_sizes: Size of the feature maps for every layer.
        :param use_batch_norm: True if batch normalization should be used or not.
        :param filter_widths: List of widths for the filter in each layer.
        :param filter_heights: List of heights for the filter in each layer. 
        :param dropout_internal: Keep probability to be used for dropout applied to hidden layers.
        :param is_training: Placeholder specifying if this is training run or not (for batch normalization)
        :param initializer: String specifying which initializer to use.
        :return: A list of outputs of each hidden layer.
        """
        # output channels of all layers
        out_channels = [input_.get_shape()[3].value] + feature_map_sizes

        # create the layers
        next_in = input_
        e_hidden_layers = []

        for i in range(len(out_channels) - 1):
            c_in = out_channels[i]
            c_out = out_channels[i + 1]
            layer_name = 'conv_l' + str(i)

            pool_factor = [2, 2]
            in_shape = [next_in.get_shape()[1].value, tf.shape(next_in)[2]]
            if in_shape[0] < 10:  # filter_heights[i] * 1.5:
                # no point in downsampling height if the height is smaller than the filter
                pool_factor[0] = 1

            # add dropout only on layers > 1
            if i != 0:
                dropout = tf.nn.dropout(next_in, keep_prob=dropout_internal)
            else:
                dropout = next_in

            conv = conv2d_strided(dropout, filter_h=filter_heights[i], filter_w=filter_widths[i],
                                  channels_in=c_in, channels_out=c_out, initializer=initializer,
                                  pool_factor=pool_factor, name=layer_name)

            if use_batch_norm:
                conv_bn = batch_normalization(conv, is_training, scope='bn_conv_layer' + str(i - 1), reuse=False)
            else:
                conv_bn = conv
            out = activation_fn(conv_bn)
            e_hidden_layers.append((out, pool_factor))
            next_in = out

        return e_hidden_layers

    @staticmethod
    def build_decoder(e_hidden_layers, activation_fn, out_activation, use_batch_norm, share_weights, filter_widths,
                      filter_heights, is_training, dropout_internal, use_unpooling, initializer, skip_connections):
        """
        Builds the decoder. The first input to the decoder is `self.latent_space`.
        :param e_hidden_layers: List of hidden layers in the encoder, first entry is original input, last entry is
          latent code, i.e. input to decoder.
        :param activation_fn: Which activation function to use for each layer (directly the Tensorflow op).
        :param out_activation: String indicating which activation function to be used on the very last layer.
        :param use_batch_norm: True if batch normalization should be used or not.
        :param share_weights: If True, learnable parameters are shared between encoder and decoder.
        :param filter_widths: List of widths for the filter in each layer.
        :param filter_heights: List of heights for the filter in each layer. 
        :param dropout_internal: Keep probability to be used for dropout applied to hidden layers.
        :param is_training: Placeholder specifying if this is training run or not (for batch normalization)
        :param use_unpooling: If True, decoder will use custom unpooling operation instead of strided deconv.
        :param initializer: String specifying which initializer to use.
        :param skip_connections: Amount of skip connections to be used by decoder.
        :return: A list of the outputs of every layer in the decoder and output of the last layer before it is activated
        """
        # e_hidden_layers is a tuple of (out shape, pool factor), unzip the two
        hidden_layers = [e_hidden_layers[0]] + [l for (l, _) in e_hidden_layers[1:]]
        pool_factors = [p for (_, p) in e_hidden_layers[1:]]

        # get list of output channels per layer in encoder (including input)
        out_channels = [l.get_shape()[3].value for l in hidden_layers]

        # get list of original output shapes per layer (including input layer)
        ori_shapes = [(l.get_shape()[1].value, tf.shape(l)[2]) for l in hidden_layers]

        next_in = hidden_layers[-1]
        d_hidden_layers = []
        last_before_activation = None
        layer_prefix = 'conv_l' if share_weights else 'deconv_l'
        for i in range(len(out_channels) - 1, 0, -1):
            c_in = out_channels[i]
            c_out = out_channels[i - 1]
            layer_name = layer_prefix + str(i - 1)

            in_shape = [next_in.get_shape()[1].value, tf.shape(next_in)[2]]
            next_shape = ori_shapes[i - 1]
            pool_factor = pool_factors[i - 1]

            if use_unpooling:
                # only do strided deconvolution in spatial domain in the last layer
                pool_factor = [2, 1]
                next_shape = (next_shape[0], in_shape[1])

            # add a dropout on input of each layer
            dropout = tf.nn.dropout(next_in, keep_prob=dropout_internal)

            deconv = deconv2d_strided(dropout, filter_h=filter_heights[i - 1], filter_w=filter_widths[i - 1],
                                      channels_in=c_in, channels_out=c_out, output_shape=next_shape,
                                      initializer=initializer, pool_factor=pool_factor,
                                      name=layer_name, reuse=share_weights)

            if use_unpooling:
                out_shape = ori_shapes[i - 1][1]
                deconv = unpool(deconv, pool_dim=2, pool_factor=2, is_training=is_training, out_shape=out_shape)

            if use_batch_norm:
                deconv_bn = batch_normalization(deconv, is_training, scope='bn_deconv_layer' + str(i - 1), reuse=False)
            else:
                deconv_bn = deconv

            a_fn = out_activation if i == 1 else activation_fn
            last_before_activation = deconv_bn
            out = a_fn(deconv_bn) if a_fn is not None else deconv_bn
            d_hidden_layers.append(out)
            next_in = out
        return d_hidden_layers, last_before_activation

    def train(self, data_feeder, n_epochs, dropout, dropout_internal,
              curriculum=None, description=None):
        """
        Default training function which drives the whole training procedure. Overwrite this method to implement custom
        training procedures.

        :param data_feeder: Provides batch-wise access to the data.
        :param n_epochs: Amount of epochs to train.
        :param dropout: Keep probability for dropout regularization (1.0 to disable).
        :param dropout_internal: Float in [0-1]. If smaller than 1, then add a dropout layer in front of each conv/deconv operation.
        :param curriculum: Curriculum to be used with ColumnPerturbator.
        :param description: Optional string describing some details about the training procedure that will go into
          the dumped configuration file.
        """
        self.prepare_training()
        self.dump_configuration(description)

        # writer for tensorboard logs
        writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())

        n_batches = data_feeder.n_batches_train()

        start_time = time.time()

        def _time_delta():
            return str(datetime.timedelta(seconds=int(time.time() - start_time)))

        best_valid_loss = float('inf')
        best_epoch = -1

        # start the session.
        with tf.Session() as sess:
            if not self.restored:
                sess.run(tf.global_variables_initializer())

            def _evaluate(global_step, visualize=False):
                # evaluate validation set
                v_loss, v_summary_str = self.evaluate_split(sess, data_feeder, split='valid')
                writer.add_summary(v_summary_str, global_step=global_step)

                # visualize random entry from validation set
                if visualize:
                    self.visualize(sess, data_feeder.random_valid_batch(np.random))

                return v_loss

            for e in range(n_epochs):
                # randomly reorder the training batches
                data_feeder.reshuffle_train()

                # update the width of the perturbator if curriculum is set
                if curriculum:
                    new_vals = curriculum.get_new_params(e)
                    self.perturbator.update(**new_vals)
                    width_set = sess.run(self.update_perturbation_width,
                                         feed_dict={self.perturbation_width_pl: new_vals['new_mean']})
                    print('Set perturbator Gaussian to N({}, {})'.format(width_set, new_vals['new_std']))

                # at the very beginning, evaluate validation set once
                if e == 0:
                    best_valid_loss = _evaluate(sess.run(self.global_step), visualize=True)
                    print('Initial validation loss: {:.8f}'.format(best_valid_loss))

                for i, batch in enumerate(data_feeder.train_batches()):
                    # training iteration for every batch
                    batch.perturbate(self.perturbator)
                    summaries_str, loss, global_step = self.train_step(sess=sess, train_op=self.train_op,
                                                                       batch=batch, dropout=dropout,
                                                                       dropout_internal=dropout_internal)

                    writer.add_summary(summaries_str, global_step=global_step)
                    print('\rEpoch: {:3d} [{:4d}/{:4d}] time: {:>8} loss: {:.8f}'.format(
                        e + 1, i + 1, n_batches, _time_delta(), loss), end='')

                    sess.run(self.update_global_step)

                # evaluate and visualize every epoch
                v_loss = _evaluate(global_step, visualize=True)

                # save if validation loss improved
                marker = ''
                if v_loss < best_valid_loss:
                    best_valid_loss = v_loss
                    best_epoch = e + 1
                    marker = ' **'
                    self.save(sess, global_step)
                print(' | validation loss: {:.8f}{}'.format(v_loss, marker))

            # remember and print best validation loss and best epoch
            bv = 'best validation loss: {:.8f}'.format(best_valid_loss)
            be = 'best epoch: {}'.format(best_epoch)
            self.append_to_configuration(bv, be)
            print(bv, be)

    def visualize(self, sess, batch):
        # choose a random datapoint in the batch
        batch_r = batch[np.random.randint(0, batch.batch_size)]
        batch_r.perturbate(self.perturbator)
        id_ = batch_r.ids[0]

        # get the model prediction of that datapoint
        loss, reconstructed = self.evaluate_batch(sess, batch_r)
        step = sess.run(self.global_step)

        plot_reconstruction(input_=batch_r.inputs_,
                            target=batch_r.targets,
                            mask=batch_r.mask,
                            reconstructions=[reconstructed],
                            title='visualizations @global step {}, id: {}, loss: {:.4f}'.format(step, id_, loss),
                            scale_foot_contacts_to=None,
                            save_path=os.path.join(self.visual_path,
                                                   'global_step{:0>6}_id{}_loss{:.1f}'.format(step, id_, loss)),
                            show=False)

    def evaluate_split(self, sess, data_feeder, split):
        total_loss = 0.0
        total_loss_l2 = 0.0
        n_data = 0
        fetch = [self.loss_op, self.l2_loss, self.reconstructed]
        for batch in data_feeder.valid_batches():
            batch.perturbate(self.perturbator)
            loss, l2_loss, _ = self.evaluate_batch(sess, batch, custom_fetch=fetch)
            batch_size = batch.batch_size
            total_loss += loss * batch_size
            total_loss_l2 += l2_loss * batch_size
            n_data += batch_size
        total_loss /= float(n_data)
        total_loss_l2 /= float(n_data)

        # store the validation loss in the graph so that we can use it for tensorboard visualizations
        sess.run([self.update_v_loss, self.update_v_l2_loss],
                 feed_dict={self.v_loss_pl: total_loss, self.v_l2_loss_pl: total_loss_l2})

        # now get the stats
        # Note: it is intential that updating and getting the stats is split in two runs. If we would do that in one
        # run, then the validation summaries might be fetched before the respective variables have been updated.
        v_summary_str = sess.run(self.valid_summaries)
        return total_loss, v_summary_str

    def evaluate_batch(self, sess, batch, custom_fetch=None):
        fetch = custom_fetch or [self.loss_op, self.reconstructed]
        feed_dict = self._create_feed_dict(batch, dropout=1.0, dropout_internal=1.0, is_training=False)
        ret = sess.run(fetch, feed_dict=feed_dict)

        # check if reconstruction was fetched and if so, convert it
        if self.reconstructed in fetch:
            idx = fetch.index(self.reconstructed)
            ret[idx] = to_1_channel_rep(ret[idx])

        return ret

    def train_step(self, sess, train_op, batch, dropout, dropout_internal):
        fetch = [self.train_summaries, self.loss_op, train_op, self.global_step]
        feed_dict = self._create_feed_dict(batch, dropout=dropout, dropout_internal=dropout_internal, is_training=True)
        [summaries_str, loss, _, global_step] = sess.run(fetches=fetch, feed_dict=feed_dict)
        return summaries_str, loss, global_step

    def _create_feed_dict(self, batch, dropout, dropout_internal, is_training):
        target_bone_lengths = compute_bone_lengths_np(batch.targets)

        # compute smoothness margin
        shifted = shift_one_frame(batch.targets)
        diff = batch.targets - shifted
        smoothness_margin = np.sqrt(np.sum(np.sum(np.multiply(diff, diff), axis=2), axis=1))

        input_ = batch.inputs_
        targets = batch.targets
        mask = batch.mask if self.use_masked_loss else np.ones(batch.inputs_.shape)

        # lay out input data in channels if required
        # this assumes that data which is fed into never uses channels
        input_shape = self.input_.get_shape()
        if len(input_shape) == 3:
            # this is an old model where input placeholder did not yet have channels, so just do nothing
            pass
        elif input_shape[3] == 1:
            # not using channels, but channel is explicit
            input_ = np.expand_dims(input_, -1)
            targets = np.expand_dims(targets, -1)
            mask = np.expand_dims(mask, -1)
        elif input_shape[3] == 3:
            # we are using channels so convert data into that format
            input_ = to_3_channels_rep(input_)
            targets = to_3_channels_rep(targets)
            mask = to_3_channels_rep(mask)
        else:
            raise RuntimeError('Shape in input placeholder is unexpected.')

        feed_dict = {self.input_: input_,
                     self.targets: targets,
                     self.mask: mask,
                     self.dropout: dropout,
                     self.dropout_internal: dropout_internal,
                     self.is_training: is_training,
                     self.bone_targets: target_bone_lengths,
                     self.smoothness_margin: smoothness_margin}
        return feed_dict


class Conv1dAutoencoder(VanillaAutoencoder):
    """
    Everything is exactly the same as for the Vanilla Autoencoder, except that the convolutional layers use 1D
    convolution over the temporal domain like Holden does.
    """

    def __init__(self, checkpoint_dir, name, perturbator, use_masked_loss, restored=False, store_to_run=None):
        super().__init__(checkpoint_dir, name, perturbator, use_masked_loss, restored, store_to_run)

    @staticmethod
    def build_encoder(input_, activation_fn, feature_map_sizes, use_batch_norm,
                      filter_widths, filter_heights, is_training, dropout_internal, initializer):
        # output channels of all layers, including the first one, which is 73
        out_channels = [input_.get_shape()[1].value] + feature_map_sizes

        # create the layers
        next_in = input_
        e_hidden_layers = []

        # filter sizes
        filter_sizes = [25, 15, 7, 3, 3]

        for i in range(len(out_channels) - 1):
            c_in = out_channels[i]
            c_out = out_channels[i + 1]
            layer_name = 'conv1d_l' + str(i)
            conv = conv1d_strided(next_in, filter_size=filter_sizes[i], channels_in=c_in, channels_out=c_out,
                                  pool_factor=2, name=layer_name)
            if use_batch_norm:
                conv_bn = batch_normalization(conv, is_training, scope=layer_name, reuse=False)
            else:
                conv_bn = conv
            out = activation_fn(conv_bn)
            e_hidden_layers.append(out)
            next_in = out

        return e_hidden_layers

    @staticmethod
    def build_decoder(e_hidden_layers, activation_fn, out_activation, use_batch_norm, share_weights, filter_widths,
                      filter_heights, is_training, dropout_internal, use_unpooling, initializer, skip_connections):
        # get list of output channels per layer in encoder (including input)
        out_channels = [l.get_shape()[3].value for l in e_hidden_layers]

        # get list of original output shapes per layer (including input layer)
        ori_shapes = [(l.get_shape()[1].value, tf.shape(l)[2]) for l in e_hidden_layers]

        next_in = e_hidden_layers[-1]
        d_hidden_layers = []
        last_before_activation = None
        for i in range(len(out_channels) - 1, 0, -1):
            c_in = out_channels[i]
            c_out = out_channels[i - 1]
            layer_name = 'conv1d_l' + str(i - 1)
            deconv = deconv1d_strided(next_in, filter_size=3, channels_in=c_in, channels_out=c_out,
                                      output_height=ori_shapes[i - 1][1], pool_factor=2, name=layer_name, reuse=True)
            if use_batch_norm:
                deconv_bn = batch_normalization(deconv, is_training, scope=layer_name, reuse=True)
            else:
                deconv_bn = deconv

            a_fn = out_activation if i == 1 else activation_fn
            last_before_activation = deconv_bn
            out = a_fn(deconv_bn)
            d_hidden_layers.append(out)
            next_in = out
        return d_hidden_layers, last_before_activation


class HoldenAutoencoder(VanillaAutoencoder):
    """
    An Autoencoder using Holden's architecture.
    """

    def __init__(self, checkpoint_dir, name, perturbator, use_masked_loss, restored=False, store_to_run=None):
        super().__init__(checkpoint_dir, name, perturbator, use_masked_loss, restored, store_to_run)

    @staticmethod
    def build_encoder(input_, activation_fn, feature_map_sizes, use_batch_norm,
                      filter_widths, filter_heights, is_training, dropout_internal, initializer):
        # note: most of the input arguments are ignored because the Holden architecture is fixed
        # note: not adding dropout on first layer, as `input_` already considers that
        conv = conv1d_strided(input_=input_,
                              filter_size=25,
                              channels_in=input_.get_shape()[1].value,
                              channels_out=256,
                              pool_factor=1,
                              name='en_conv1d')
        act = activation_fn(conv)
        pooled = tf.nn.max_pool(act, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
        return [pooled]

    @staticmethod
    def build_decoder(e_hidden_layers, activation_fn, out_activation, use_batch_norm, share_weights, filter_widths,
                      filter_heights, is_training, dropout_internal, use_unpooling, initializer, skip_connections):
        unpooled = unpool(e_hidden_layers[-1], pool_dim=2, pool_factor=2, is_training=is_training)
        drop = tf.nn.dropout(unpooled, dropout_internal)
        conv = conv1d_strided(input_=drop,
                              filter_size=25,
                              channels_in=drop.get_shape()[1].value,
                              channels_out=e_hidden_layers[0].get_shape()[1].value,
                              pool_factor=1,
                              name='de_conv1d')
        act = out_activation(conv) if out_activation is not None else conv
        return [act], None


class VGGAutoencoder(VanillaAutoencoder):
    CHANNELS_AND_REP = [(32, 2), (64, 2), (128, 2), (256, 2), (256, 2)]

    def __init__(self, checkpoint_dir, name, perturbator, use_masked_loss, restored=False, store_to_run=None):
        super().__init__(checkpoint_dir, name, perturbator, use_masked_loss, restored, store_to_run)

    @staticmethod
    def build_encoder(input_, activation_fn, feature_map_sizes, use_batch_norm,
                      filter_widths, filter_heights, is_training, dropout_internal, initializer):
        # `feature_map_sizes` is ignored because this is fixed in VGG models
        # `filter_widths` and `filter_heights` ignored at the moment
        # `input_` has shape (batch_size, height, width, channels)
        filter_h = filter_w = 3

        # return last output of each layer
        e_hidden_layers = []

        def _build_layer(input_, n_reps, channels_out, name, filter_h, filter_w):
            next_in = input_
            for i in range(n_reps):
                c_in = next_in.get_shape()[3].value
                layer_name = name + '_conv' + str(i)
                out = conv2d_strided(next_in, filter_h=filter_h, filter_w=filter_w,
                                     channels_in=c_in, channels_out=channels_out, initializer=initializer,
                                     pool_factor=(1, 1), name=layer_name)
                if use_batch_norm:
                    out_bn = batch_normalization(out, is_training, scope='{}_bn_conv_rep{}'.format(name, i),
                                                 reuse=False)
                else:
                    out_bn = out
                next_in = activation_fn(out_bn)
            return next_in

        next_in = input_
        for i, (channels_out, rep) in enumerate(VGGAutoencoder.CHANNELS_AND_REP):
            name = 'layer' + str(i)
            with tf.name_scope(name):
                out = _build_layer(next_in, n_reps=rep, channels_out=channels_out, name=name,
                                   filter_h=filter_h, filter_w=filter_w)

            name = 'pool' + str(i)
            with tf.name_scope(name):
                # do not pool spatial domain if it is too low
                spatial_factor = 2 if out.get_shape()[1] >= 3 else 1
                pool_factor = [spatial_factor, 2]
                pooled = tf.nn.max_pool(out, ksize=[1, pool_factor[0], pool_factor[1], 1],
                                        strides=[1, pool_factor[0], pool_factor[1], 1], padding='SAME')

            # append spatial factor so that decoder knows if it should uspample or not
            e_hidden_layers.append((pooled, pool_factor))
            next_in = pooled

        return e_hidden_layers

    @staticmethod
    def build_decoder(e_hidden_layers, activation_fn, out_activation, use_batch_norm, share_weights, filter_widths,
                      filter_heights, is_training, dropout_internal, use_unpooling, initializer, skip_connections):
        # `share_weights`, `dropout_internal` and `use_unpooling` is ignored but this is not going to chagne
        filter_h = filter_w = 3
        channels_and_rep = VGGAutoencoder.CHANNELS_AND_REP

        # e_hidden_layers is list of tuples, unzip this
        hidden_layers = [e_hidden_layers[0]] + [l for (l, _) in e_hidden_layers[1:]]
        pool_factors = list(reversed([p for (_, p) in e_hidden_layers[1:]]))

        # get list of original output shapes per layer (including input layer)
        ori_shapes = [(l.get_shape()[1].value, tf.shape(l)[2]) for l in hidden_layers]
        ori_shapes = list(reversed(ori_shapes))

        def _build_layer(input_, n_reps, last_out_channel, output_shape, pool_factor, name, is_last=False):
            next_in = input_
            for i in range(n_reps):
                c_in = next_in.get_shape()[3].value
                c_out = last_out_channel if i == n_reps - 1 else c_in
                layer_name = name + '_deconv' + str(i)

                if i == 0:
                    # upsample in the first deconv layer
                    pool_factor_ = pool_factor
                else:
                    pool_factor_ = [1, 1]

                # deconvolution
                out = deconv2d_strided(next_in, filter_h=filter_h, filter_w=filter_w,
                                       channels_in=c_in, channels_out=c_out, initializer=initializer,
                                       pool_factor=pool_factor_, name=layer_name, output_shape=output_shape)

                # batch normalization
                if use_batch_norm:
                    out_bn = batch_normalization(out, is_training, scope='{}_bn_deconv_rep{}'.format(name, i),
                                                 reuse=False)
                else:
                    out_bn = out

                # activation
                if is_last and i == n_reps - 1:
                    next_in = out_activation(out_bn) if out_activation is not None else out_bn
                else:
                    next_in = activation_fn(out_bn)

            return next_in

        d_hidden_layers = []
        next_in = hidden_layers[-1]
        target_channels = [hidden_layers[0].get_shape()[-1].value] + [v[0] for v in channels_and_rep[:-1]]
        target_channels = list(reversed(target_channels))
        n_hidden_layers = len(hidden_layers)
        for i, (_, rep) in enumerate(reversed(channels_and_rep)):
            name = 'layer' + str(i)
            with tf.name_scope(name):
                if 1 <= i <= skip_connections:
                    # use skip connections starting from 2nd hidden layer in encoder
                    encoder_out = hidden_layers[n_hidden_layers - i - 1]

                    # concatenate with input
                    next_in = tf.concat(3, [next_in, encoder_out])

                out = _build_layer(next_in, n_reps=rep, last_out_channel=target_channels[i],
                                   output_shape=ori_shapes[i + 1], pool_factor=pool_factors[i],
                                   name=name, is_last=(i == len(channels_and_rep) - 1))

            d_hidden_layers.append(out)
            next_in = out

        return d_hidden_layers, None


class InceptionAutoencoder(VGGAutoencoder):
    """
    Subclassing VGGAutoencoder so that we can reuse the same decoder.
    """

    def __init__(self, checkpoint_dir, name, perturbator, use_masked_loss, restored=False, store_to_run=None):
        super().__init__(checkpoint_dir, name, perturbator, use_masked_loss, restored, store_to_run)

    @staticmethod
    def build_encoder(input_, activation_fn, feature_map_sizes, use_batch_norm,
                      filter_widths, filter_heights, is_training, dropout_internal, initializer):

        def _build_inception_layer(input_, n_branches, n_reps, n_channels, filter_size, name):
            branch_outs = []
            for i in range(n_branches):
                # create the branches
                next_in = input_

                for j in range(n_reps):
                    # each branch might have several layers
                    c_in = next_in.get_shape()[-1].value
                    c_out = n_channels
                    layer_name = name + '_branch{}_conv{}'.format(i, j)
                    out = conv2d_strided(next_in,
                                         filter_h=filter_size + 2 * i,
                                         filter_w=filter_size + 2 * i,
                                         channels_in=c_in,
                                         channels_out=c_out,
                                         initializer=initializer,
                                         pool_factor=1,
                                         name=layer_name)
                    next_in = activation_fn(out)

                branch_outs.append(next_in)

            # concatenate all branches depthwise
            out = tf.concat(3, branch_outs)
            return out

        channels = [8, 32, 64, 64, 64]  # channel per branch, final number of channels is this times `n_branches`
        next_in = input_
        e_hidden_layers = []
        for i, n_channels in enumerate(channels):
            name = 'inception{}'.format(i)
            with tf.name_scope(name):
                out = _build_inception_layer(next_in,
                                             n_branches=4,
                                             n_reps=2,
                                             n_channels=n_channels,
                                             filter_size=3,
                                             name=name)

            with tf.name_scope('pool{}'.format(i)):
                pooled = tf.nn.max_pool(out,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME')

            next_in = pooled
            e_hidden_layers.append(pooled)

        return e_hidden_layers
