import os
import tensorflow as tf
import numpy as np
import flags_parser

from tbase import utils
from tbase.data_loader import LoaderV2
from tbase.data_loader import FeederV2
from tbase.visualizer import Visualizer
from tbase.skeleton import SkeletonSequence
from tbase.skeleton import Skeleton
from tbase.normalizers import Normalizer
from models import VanillaAutoencoder
from models import compute_bone_lengths_np
from models import shift_one_frame
from plotter import plot_reconstruction
from plotter import show_np_arrays


fs = tf.app.flags
flags_parser.define_perturbator(fs, 'none')
flags_parser.define_perturbation_amount(fs, 15)
flags_parser.define_perturbation_size(fs, '[2, 4, 6]')
flags_parser.define_seed(fs, 42)
flags_parser.define_if_test(fs)
flags_parser.define_if_masked_loss(fs)

fs.DEFINE_string('model1', 'none', 'which model to visualize, expected format is "model_name/run_id[f]" [none]')
fs.DEFINE_string('model2', 'none', 'which model to compare to, same format as --model1 [none]')
fs.DEFINE_integer('samples', 10, 'how many samples to visualize [10]')
fs.DEFINE_string('show_id', 'none', 'show datapoint(s) of the given (comma seperated) id(s) or random validation point if none [none]')
fs.DEFINE_boolean('show_images', False, 'if set, visualizes images instead of the motion')
fs.DEFINE_boolean('show_smoothness', False, 'if set, visualizes the smoothness when --show_images is set')
fs.DEFINE_boolean('replace_unmasked', False, 'if set, replaces the prediction belonging to the non-corrupted part with the original, only when --show_images')
fs.DEFINE_boolean('overlay', False, 'if set, the output from --model2 is drawn on top of output from --model1')
fs.DEFINE_integer('overlay_ori', -1, 'if >= 0, the original motion is overlain with the selected run [-1]')
fs.DEFINE_string('descr1', 'model 1', 'description string displayed for model given in --model1 [model 1]')
fs.DEFINE_string('descr2', 'model 2', 'description string displayed for model given in --model2 [model 2]')
fs.DEFINE_bool('hide_floor', False, 'if set, no floor is drawn')
fs.DEFINE_string('export_to_csv', 'none', 'filename to export motion to or "none" [none]')


FLAGS = fs.FLAGS
RNG = np.random.RandomState(FLAGS.seed)


def _compute_l2(x, y):
    """
    Compute squared L2 error between matrices `x` and `y`. Matrices can be batched. In this case, the L2 error per
    batch is returned.
    """
    assert x.shape == y.shape
    diff = x - y
    l2_err = np.sum(np.sum(np.multiply(diff, diff), axis=-1), axis=-1)
    return l2_err


def _print_bone_lengths(ori, reconstruction):
    """
    Print some bone length statistics. Input is expected to be of shape [dim, seq_length].
    """
    ori_lengths = compute_bone_lengths_np(ori[np.newaxis, :])
    rec_lengths = compute_bone_lengths_np(reconstruction[np.newaxis, :])

    def _compute_bone_length_stats(x):
        x = np.squeeze(x, 0) if x.shape[0] == 1 and len(x.shape) == 3  else x
        means = np.mean(x, axis=1)
        stds = np.std(x, axis=1)
        return means, stds

    ori_mean, ori_std = _compute_bone_length_stats(ori_lengths)
    rec_mean, rec_std = _compute_bone_length_stats(rec_lengths)
    diffs = np.fabs(ori_mean - rec_mean)
    stats = np.transpose(np.array([ori_mean, ori_std, rec_mean, rec_std, diffs]), [1, 0])
    print('ori_mean ori_std rec_mean rec_std abs_diff')
    for om, os, rm, rs, d in stats:
        print('{:8.3f} {:7.4f} {:8.3f} {:7.4f} {:8.3f}'.format(om, os, rm, rs, d))

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    print('mean of absolute differences: {:.4f} (+/- {:.4f})'.format(mean_diff, std_diff))


def _maybe_replace_unmasked(ori, mask, reconstruction):
    """
    Replace the unmasked parts of `reconstruction` with the respective parts from `ori` if configured in `FLAGS`. 
    """
    mask_b = np.array(mask, dtype=np.bool)
    # if mask consists only of False, it means it is empty, in which case we don't want to replace anything.
    if len(np.where(mask_b == True)[0]) == 0:
        return reconstruction
    if FLAGS.replace_unmasked:
        mask_inv = ~mask_b
        reconstruction[mask_inv] = ori[mask_inv]
    return reconstruction


def _compute_smoothness(data):
    """
    Computes a smoothness measure for the input data by computing the difference in poses between consecutive frames.
    :param data: A np array of size (n_batches, dim, seq_length)
    :return: A np array of the same size as `data`
    """
    ori_ndim = len(data.shape)
    x = data[np.newaxis, ...] if ori_ndim == 2 else data
    shifted = shift_one_frame(x)

    # difference of ori to shifted is the measure of smoothness
    diff = x - shifted
    diff = np.squeeze(diff, axis=0) if ori_ndim == 2 else diff
    return diff


def _show_smoothness(data, titles):
    assert isinstance(data, list)
    assert len(data) == len(titles)

    # compute smoothness for all inputs
    data_s = []
    for i, d in enumerate(data):
        d_smoothed = _compute_smoothness(d)
        d_norm = np.sqrt(np.sum(np.multiply(d_smoothed, d_smoothed)))
        data_s.append(d_smoothed)
        titles[i] += ', smoothness: {:.4f}'.format(d_norm)

    # scale all inputs the [-1, 1] so that we can make a fair comparison
    x_min = np.amin([np.amin(x) for x in data_s])
    x_max = np.amax([np.amax(x) for x in data_s])
    x_div = x_max - x_min
    data_s_scaled = [(d - x_min) / x_div * 2.0 - 1.0 for d in data_s]
    show_np_arrays(data_s_scaled, 'Smoothness comparison', titles)


def show_images(batch, reconstruction, l2_losses, reconstruction_c=None, l2_losses_c=None):
    """
    Visualize the reconstruction as images like it is done during training.
    """
    reconstructions = [reconstruction]
    sub_titles = [FLAGS.descr1]
    losses = [l2_losses]

    if reconstruction_c is not None:
        reconstructions.append(reconstruction_c)
        sub_titles.append(FLAGS.descr2)
        losses.append(l2_losses_c)

    title_add = ', uncorrupted portion replaced with original' if FLAGS.replace_unmasked else ''

    for i, (in_, ori, id, mask) in enumerate(batch.all_entries()):
        if FLAGS.show_smoothness:
            _show_smoothness([ori] + [r[i] for r in reconstructions],
                             titles=['original'] + sub_titles)

        recns = []
        titles = []
        for j, r in enumerate(reconstructions):
            recn = _maybe_replace_unmasked(ori, mask, r[i])
            titles.append(sub_titles[j] + ', squared L2 loss {:.4f}'.format(losses[j][i]))
            recns.append(recn)

        plot_reconstruction(input_=in_,
                            target=ori,
                            mask=mask,
                            reconstructions=recns,
                            title='id: {}{}'.format(id, title_add),
                            scale_foot_contacts_to=(0.0, 1.0),
                            sub_titles=titles,
                            save_path=None,
                            show=True)


def show_motion(batch, reconstruction, l2_losses, reconstruction_c=None, l2_losses_c=None, caption='',
                replace_trajectory=False, replace_trajectory_c=False):
    """
    Visualize the reconstruction as 3D motion. Note that the input data is assumed to be unnormalized already!
    """
    motion_recn = reconstruction
    motion_recn_c = reconstruction_c if reconstruction_c is not None else None

    for i, (_, original, id_, mask) in enumerate(batch.all_entries()):
        motion_ori = original

        # replace original trajectory
        n_coords = len(Skeleton.ALL_JOINTS)*3
        if replace_trajectory:
            motion_recn[i, n_coords:n_coords+3] = motion_ori[n_coords:n_coords+3]
        if motion_recn_c is not None and replace_trajectory_c:
            motion_recn_c[i, n_coords:n_coords+3] = motion_ori[n_coords:n_coords+3]

        # replace unmasked part if necessary
        recn = _maybe_replace_unmasked(motion_ori, mask, motion_recn[i])
        recn_c = None if motion_recn_c is None else _maybe_replace_unmasked(motion_ori, mask, motion_recn_c[i])

        # find out which frames were masked to visualize it
        idxs = np.where(~(mask < 1e-6).all(axis=0))[0]
        idxs = idxs.flatten()

        vi = Visualizer(caption=caption, show_floor=not FLAGS.hide_floor)
        pos_ori = 0.0
        pos_one = pos_ori if FLAGS.overlay_ori == 0 else 30.0

        sks = [SkeletonSequence(motion_ori,
                                x_offset=pos_ori, name='Original (id: {})'.format(id_)),
               SkeletonSequence(recn,
                                x_offset=pos_one,
                                name='Reconstructed ({}, squared L2 loss: {:.4f})'.format(FLAGS.descr1,
                                                                                          l2_losses[i]),
                                interp=idxs)]
        if recn_c is not None:
            pos_two = pos_ori if FLAGS.overlay_ori == 1 else pos_one + 30
            pos_two = pos_one if FLAGS.overlay else pos_two
            sks.append(SkeletonSequence(recn_c,
                                        x_offset=pos_two,
                                        name='Reconstructed ({}, squared L2 loss: {:.4f})'.format(FLAGS.descr2,
                                                                                                  l2_losses_c[i]),
                                        interp=idxs))

        if FLAGS.export_to_csv.lower() != 'none':
            [sk.export_to_csv(FLAGS.export_to_csv.replace('.txt', '_{}.txt').format(i)) for i, sk in enumerate(sks)]

        vi.load_skeleton_sequences(sks)
        vi.show_()


def main(argv):
    if len(argv) > 1:
        # we have some unparsed flags
        raise ValueError('unknown flags: {}'.format(' '.join(argv[1:])))

    # get the perturbator to use
    perturbator = flags_parser.get_perturbator(FLAGS, is_inference=True)

    # load the data without normalizing it because this is done manually per batch
    data_path = flags_parser.get_data_path()
    loader = LoaderV2(train_path=os.path.join(data_path, 'train'),
                      valid_path=os.path.join(data_path, 'valid'),
                      normalizer=None,  # is done manually per batch
                      discard_foot_contacts=False)  # is done manually per batch
    data = loader.get_validation_unnormalized_all()
    feeder = FeederV2(data_train=None,
                      data_valid=data,
                      batch_size=FLAGS.samples,
                      rng=RNG)

    # get the test batch
    if not FLAGS.show_id.lower() == 'none':
        batch = feeder.valid_batch_from_idxs(flags_parser.string_to_list(FLAGS.show_id))
    else:
        batch = feeder.random_valid_batch(RNG)

    def _get_model_path(checkpoint_dir, name, run):
        return os.path.join(checkpoint_dir, name, 'run_{:0>3}'.format(run))

    def _get_normalizer(model_path):
        normalizer = Normalizer.from_disk(model_path)
        normalizer.load(model_path)
        return normalizer

    def _normalize_batch(normalizer, batch):
        b = batch.copy()
        b.inputs_ = normalizer.normalize(b.inputs_)
        b.targets = normalizer.normalize(b.targets)
        return b

    def _get_prediction(model_path, batch, reapply_perturbation, discard_foot_contacts):
        # discard foot contact information if necessary:
        if discard_foot_contacts:
            batch_r = batch.copy()
            batch_r.remove_foot_contacts()
        else:
            batch_r = batch

        # normalize batch
        normalizer = _get_normalizer(model_path)
        batch_normalized = _normalize_batch(normalizer, batch_r)

        # show_np_arrays(batch_normalized.targets, 'bla', [''] * batch_normalized.targets.shape[0])

        # introduce corruption
        batch_normalized.perturbate(perturbator, reapply=reapply_perturbation)

        # restore the graph
        tf.reset_default_graph()
        latest_checkpoint = utils.get_latest_checkpoint(model_path)
        if latest_checkpoint is None:
            raise RuntimeError('could not find checkpoint {}'.format(model_path))

        print('restoring checkpoint {} ...'.format(latest_checkpoint))
        model = VanillaAutoencoder.build_from_metagraph(checkpoint=latest_checkpoint,
                                                        perturbator=perturbator,
                                                        use_masked_loss=FLAGS.use_masked_loss)

        # compute reconstruction, only use fraction of available GPU memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # restore the model
            model.load(sess, latest_checkpoint)

            # get the prediction
            loss, recn = model.evaluate_batch(sess, batch_normalized)
            print('reconstruction error: {:.4f}'.format(loss))

        # unnormalize
        recn_un = normalizer.unnormalize(recn)

        # compute the L2 loss for every data point in the batch
        l2_losses = []
        for i, (_, original, _, _) in enumerate(batch_normalized.all_entries()):
            l2_losses.append(_compute_l2(original, recn[i]))

        return recn, recn_un, l2_losses

    # get the prediction of the model
    model_name, run_id, discard_foot_contacts, replace_traj = utils.extract_model_name_and_run(FLAGS.model1)
    model_path = _get_model_path(flags_parser.get_checkpoints_path(), model_name, run_id)
    recn, recn_un, l2_losses = _get_prediction(model_path, batch,
                                               reapply_perturbation=False,
                                               discard_foot_contacts=discard_foot_contacts)

    # if necessary, compute second reconstruction to compare
    replace_traj_c = False
    if FLAGS.model2.lower() != 'none':
        model_name, run_id, discard_foot_contacts, replace_traj_c = utils.extract_model_name_and_run(FLAGS.model2)
        model_path = _get_model_path(flags_parser.get_checkpoints_path(), model_name, run_id)
        recn_c, recn_c_un, l2_losses_c = _get_prediction(model_path, batch,
                                                         reapply_perturbation=True,
                                                         discard_foot_contacts=discard_foot_contacts)
    else:
        recn_c = recn_c_un = l2_losses_c = None

    # batch.remove_foot_contacts()

    # Re-apply perturbation to original for visualization.
    # If foot contacts were discarded previously, this batch here still has shape (batch_size 73) so the mask for
    # the re-applicaiton will not consider the foot contacts. But because this is just for visualization, we don't
    # care at the moment.
    batch.perturbate(perturbator, reapply=True)

    # visualize
    if FLAGS.show_images:
        show_images(batch, recn_un,l2_losses, recn_c_un, l2_losses_c)
    else:
        vs = 'vs. {}'.format(FLAGS.model2) if FLAGS.model2.lower() != 'none' else ''
        caption = 'model {} {}'.format(FLAGS.model1, vs)
        show_motion(batch, recn_un, l2_losses, recn_c_un, l2_losses_c, caption,
                    replace_trajectory=replace_traj, replace_trajectory_c=replace_traj_c)


if __name__ == '__main__':
    tf.app.run()
