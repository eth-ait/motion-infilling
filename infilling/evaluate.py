import os
import flags_parser
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tbase.utils as utils

from collections import OrderedDict
from tabulate import tabulate
from models import compute_bone_lengths_np
from models import VanillaAutoencoder
from tbase.normalizers import Normalizer
from tbase.data_loader import LoaderV2
from tbase.data_loader import FeederV2
from tbase.skeleton import to_global_batched
from tbase.skeleton import Skeleton


fs = tf.app.flags
flags_parser.define_perturbator(fs, 'none')
flags_parser.define_perturbation_amount(fs, 1)
flags_parser.define_perturbation_size(fs, '[60]')
flags_parser.define_seed(fs, 42)
flags_parser.define_if_test(fs)
flags_parser.define_batch_size(fs, 80)

fs.DEFINE_string('split', 'validation', 'which data split to evaluate "training", "validation" or "test" [validation]')
fs.DEFINE_string('runs', 'VanillaCAE/0', 'which run(s) to evaluate, accepted format: "model1_name/run1,run2:model2_name/run1,run2" [VanillaCAE/0]')
fs.DEFINE_string('headers', 'none', 'header names for --runs used when displaying the stats [none]')
fs.DEFINE_boolean('lerp', False, 'Run linear interpolation baseline.')

FLAGS = fs.FLAGS
RNG = np.random.RandomState(FLAGS.seed)

MAIN_NAME = 'main'
BONE_NAME = 'bone length'
SMOOTH_NAME = 'smoothness'
TOTAL_NAME = 'total'
L2_NAME = 'l2'


def _extract_runs(runs):
    models = [(m.split('/')[0], m.split('/')[1].split(',')) for m in runs.split(';')]
    return models


def _analyze_loss(summary_str, eta, alpha, correct_loss=False):
    """
    Extracts more detailed information about the loss value from the given summary string. This is kind of an ugly
    function because it must be backwards-compatible and because it works with summary strings in the first place.
    Should be unnecessary later when the models are set up correctly.
    
    :param summary_str: A summary string, i.e. the output of sess.run(model.train_summaries).
    :param eta: scaling weight applied to bone length loss
    :param alpha: scaling weight applied to smoothness loss
    :param correct_loss: If True, additional loss terms are corrected (if used at all) because older models did not
      calculate the loss in the same way as newer models.
    :return: The main-, bone-length-, smoothness- and total-loss. Also separately the L2 loss if it is stored in the
      model.
    """
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary_str)

    # find out which loss summaries are available
    summs = {}
    for val in summary_proto.value:
        tag = val.tag
        if tag.startswith('train_summaries'):
            name = tag.split('/')[1]
            summs[name] = val.simple_value

    # map the losses found to main-, bone-length-, smoothness- an total-loss
    m_loss = b_loss = s_loss = t_loss = 0.0
    l_key, m_key, b_key, s_key, t_key = 'loss', 'main_loss', 'bone_length_loss', 'smoothness_loss', 'total_loss'
    names = summs.keys()
    if l_key in names:
        # early model version where we only had one loss value
        m_loss = summs[l_key]
        t_loss = summs[l_key]

        if b_key in names:
            b_loss = summs[b_key]
            m_loss = t_loss - b_loss

    else:
        # if 'loss' not available it was splitted to 'main_loss' and 'total_loss'
        m_loss = summs[m_key]
        t_loss = summs[t_key]

        if b_key in names:
            b_loss = summs[b_key]
        if s_key in names:
            s_loss = summs[s_key]

    if correct_loss:
        if eta > 0.0 and b_key in names:
            b_loss /= eta
        if alpha > 0.0 and s_key in names:
            s_loss /= alpha

    l2_loss = summs['g_loss_l2'] if 'g_loss_l2' in names else None

    return (m_loss, b_loss, s_loss, t_loss), l2_loss


def _compute_l2(batch, recns):
    l2_losses = []
    for i, (_, original, _, _) in enumerate(batch.all_entries()):
        diff = original - recns[i]
        l2_err = np.sum(np.sum(np.multiply(diff, diff), axis=-1), axis=-1)
        l2_losses.append(l2_err)
    return l2_losses


def _get_stats_for_model(model_path, unnormalized_data, discard_foot_contacts, use_perturbator):

    # get the perturbator
    perturbator = flags_parser.get_perturbator(FLAGS, is_inference=True) if use_perturbator else None

    if discard_foot_contacts:
        data = np.delete(unnormalized_data, obj=list(range(unnormalized_data.shape[1] - 4,
                                                           unnormalized_data.shape[1])), axis=1)
    else:
        data = unnormalized_data

    # get the normalizer
    normalizer = Normalizer.from_disk(model_path)
    normalizer.load(model_path)

    # normalize the data (this might be different depending on the model)
    data = normalizer.normalize(data.squeeze())

    # create streaming access to data with pre-defined splits
    feeder = FeederV2(data_train=None,
                      data_valid=data,
                      batch_size=FLAGS.batch_size)

    # restore the graph
    tf.reset_default_graph()
    latest_checkpoint = utils.get_latest_checkpoint(model_path)
    if latest_checkpoint is None:
        raise RuntimeError('could not find checkpoint {}'.format(model_path))
    print('restoring checkpoint {} ...'.format(latest_checkpoint))

    model = VanillaAutoencoder.build_from_metagraph(checkpoint=latest_checkpoint,
                                                    perturbator=perturbator,
                                                    use_masked_loss=False)
    # get the data split
    s = FLAGS.split.lower()
    if s == 'training':
        batches, n_batches = feeder.train_batches, feeder.n_batches_train()
    elif s == 'validation':
        batches, n_batches = feeder.valid_batches, feeder.n_batches_valid()
    elif s == 'test':
        batches, n_batches = feeder.test_batches, feeder.n_batches_test()
    else:
        raise ValueError('split "{}" unknown'.format(s))

    # get the scaling weights for the different parts of the loss function
    alpha = utils.get_config_entry(model_path, 'alpha')
    alpha = float(alpha) if alpha else 0.0
    eta = utils.get_config_entry(model_path, 'eta')
    eta = float(eta) if eta else 0.0

    # do the evaluation
    with tf.Session() as sess:
        # restore all the variables
        model.load(sess, latest_checkpoint)

        # collect losses so that we can compute mean and std later
        tot_losses = []
        all_losses = []
        all_joint_errors = []
        all_bone_lengths = []
        l2_loss = 0.0
        n_data = 0.0

        # loop over all batches
        for i, b in enumerate(batches()):
            print('\revaluating batch [{:>4} / {:>4}]'.format(i + 1, n_batches), end='')

            # keep track of batchsize
            batch_size = b.batch_size
            n_data += batch_size

            # introduce corruption
            b.perturbate(perturbator)

            fetch = [model.loss_op, model.reconstructed, model.train_summaries]
            tot_loss, recn, summaries = model.evaluate_batch(sess, b, custom_fetch=fetch)
            tot_losses.append(tot_loss * batch_size)

            # parse summaries to extract more details about loss
            all_loss, l2_loss_b = _analyze_loss(summaries, eta, alpha)
            all_losses.append(np.array(all_loss) * batch_size)

            # compute squared l2 loss
            l2_loss += np.sum(_compute_l2(b, recn)) if l2_loss_b is None else l2_loss_b * batch_size

            # Convert to 3D positions for comparisons.
            pred_un = normalizer.unnormalize(recn)
            targets = normalizer.unnormalize(b.targets)

            if FLAGS.lerp:
                # Replace predictions with linear interpolation from last to first known frame.
                joint_error_infilled = []
                if b.mask.max() > 0.0:
                    # This is assuming missing frames are consecutive.
                    idxs = np.where(b.mask[:, 0] >= 1.0)[-1].reshape(batch_size, -1)  # (batch_size, gap_size)
                    targets_global = to_global_batched(targets)  # (N, n_joints, 3, seq_len)
                    for bb in range(batch_size):
                        sf = idxs[bb, 0] - 1
                        ef = idxs[bb, -1] + 1
                        gap_size = len(idxs[bb])
                        v0 = targets_global[bb, :, :, sf].reshape(-1)
                        v1 = targets_global[bb, :, :, ef].reshape(-1)
                        ts = np.linspace(0.0, 1.0, gap_size + 2)[np.newaxis, 1:-1]  # don't want x and y to be reproduced, so +2
                        v0 = np.tile(v0[:, np.newaxis], (1, gap_size))
                        v1 = np.tile(v1[:, np.newaxis], (1, gap_size))
                        interpolated = v0 * (1 - ts) + ts * v1  # (dof, gap_size)

                        prediction = interpolated.reshape(-1, 3, gap_size)
                        gt = targets_global[bb, :, :, idxs[bb, 0]:idxs[bb, -1]+1]
                        joint_error = np.linalg.norm(prediction - gt, axis=1)
                        joint_error_infilled.append(joint_error.T)

                all_joint_errors.append(joint_error_infilled)
            else:
                # Override with target trajectory so that joint comparison is fair.
                body_dim = len(Skeleton.ALL_JOINTS) * 3
                traj = targets[:, body_dim:body_dim+3]
                pred_global = to_global_batched(pred_un, override_trajectory=traj, override_root=targets[:, 0:3])  # (N, n_joints, 3, seq_len)
                targets_global = to_global_batched(targets)  # (N, n_joints, 3, seq_len)

                # Compute 3D reconstruction error per joint and frame
                joint_error = np.linalg.norm(pred_global - targets_global, axis=2)

                joint_error = np.transpose(joint_error, [0, 2, 1])  # (N, seq_len, n_joints)
                if b.mask.max() > 0.0:
                    infilled_joint_error_only = []
                    # We had some masking going on - find indices of masked frames.
                    idxs = np.where(b.mask[:, 0] >= 1.0)[-1].reshape(batch_size, -1)  # (batch_size, gap_size)
                    for bb in range(batch_size):
                        infilled_joint_error_only.append(joint_error[bb, idxs[bb]])
                    joint_error = np.array(infilled_joint_error_only)

                all_joint_errors.append(joint_error)

            # Compute bone lengths.
            seq_len = pred_global.shape[-1]
            bone_lengths_pred = compute_bone_lengths_np(pred_global.reshape(batch_size, -1, seq_len))
            all_bone_lengths.append(bone_lengths_pred.transpose([0, 2, 1]))  # (N, N_FRAMES, N_BONES)

        # compute stats
        tot_loss_mean = np.sum(tot_losses) / float(n_data)
        all_loss_mean = np.sum(all_losses, axis=0) / float(n_data)
        l2_loss_mean = l2_loss / float(n_data)

        assert abs(tot_loss_mean - all_loss_mean[-1]) < 1e-8

        # Compute 3D joint reconstruction loss.
        all_joint_errors = np.row_stack(all_joint_errors)  # (n, seq_len, n_joints)
        joint_error_mean = np.mean(all_joint_errors[:, :, 1:])  # ignore the root
        joint_error_std = np.std(all_joint_errors[:, :, 1:])
        print()
        print(joint_error_mean, joint_error_std, '[cm]')

        # Compute bone length stats.
        # np.savez_compressed(os.path.join(model_path, 'bone_lengths.npz'), bone_lengths=all_bone_lengths)
        all_bone_lengths = np.row_stack(all_bone_lengths).reshape((-1, len(Skeleton.BONES)))  # (n*seq_len, n_bones)
        bone_means = np.mean(all_bone_lengths, axis=0)
        bone_stds = np.std(all_bone_lengths, axis=0)
        bone_std_vs_mean = bone_stds * 100.0 / bone_means

        print('Bone Lengths mean / std / perc[cm]')
        print(np.column_stack([bone_means, bone_stds, bone_std_vs_mean]))
        plt.boxplot(all_bone_lengths, showfliers=False)
        plt.ylim([0, 55])
        plt.ylabel('Bone Length [cm]')
        plt.show()

        return {MAIN_NAME: all_loss_mean[0],
                BONE_NAME: all_loss_mean[1],
                SMOOTH_NAME: all_loss_mean[2],
                TOTAL_NAME: all_loss_mean[3],
                L2_NAME: l2_loss_mean}


def main(argv):
    if len(argv) > 1:
        # we have some unparsed flags
        raise ValueError('unknown flags: {}'.format(' '.join(argv[1:])))

    # get all the runs we need to evaluate
    model_runs = _extract_runs(FLAGS.runs)
    n_runs = np.sum([len(m[1]) for m in model_runs])

    # make sure supplied headers match
    if FLAGS.headers.lower() != 'none':
        custom_header = FLAGS.headers.split(',')
        assert len(custom_header) == n_runs, 'must supply as many headers as there are runs'
        use_header = True
    else:
        use_header = False

    # load the data from disk but do not normalize as normalization is model-dependent
    data_path = flags_parser.get_data_path()
    loader = LoaderV2(train_path=os.path.join(data_path, 'train'),
                      valid_path=os.path.join(data_path, 'valid'),
                      normalizer=None,  # is done manually per batch
                      discard_foot_contacts=False)  # is done manually per batch
    unnormalized_data = loader.get_validation_unnormalized_all()

    # evaluate all the models
    stats_to_print = [(MAIN_NAME, []), (BONE_NAME, []), (SMOOTH_NAME, []), (TOTAL_NAME, []), (L2_NAME, [])]
    stats_to_print = OrderedDict(stats_to_print)
    header = ['loss']

    for (model_name, runs) in model_runs:
        for run in runs:
            # parse the run info
            discard_foot_contacts = False
            use_perturbator = False
            remove = 0
            if 'f' in run:
                discard_foot_contacts = True
                remove += 1
            if 'p' in run:
                use_perturbator = True
                remove += 1
            run_id = int(run[:-remove]) if remove > 0 else int(run)
            # compile the model path
            model_path = flags_parser.create_model_path(flags_parser.get_checkpoints_path(),
                                                        model_name, run_id)

            # compute the stats
            stats = _get_stats_for_model(model_path, unnormalized_data, discard_foot_contacts, use_perturbator)
            [stats_to_print[k].append(stats[k]) for k in stats_to_print.keys()]

            header.append('{}/{}'.format(model_name, run))

    # print the stats in nice tabular form
    header[1:] = custom_header if use_header else header[1:]
    content = [[k] + v for k, v in stats_to_print.items()]
    print('\n')
    print(tabulate(content, headers=header))


if __name__ == '__main__':
    tf.app.run()
