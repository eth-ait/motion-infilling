"""
Denoise the input and reconstruct.
"""
from showcase_utils import *
from tbase.skeleton import Skeleton
from plotter import show_np_arrays

fs = tf.app.flags
fs.DEFINE_string('model', 'none', 'which model to use for the prediction, expected format is "model_name/run_id[ft]" [none]')
fs.DEFINE_integer('samples', 3, 'amount of examples to be shown')

FLAGS = fs.FLAGS


def add_random_zeros(batch):
    rng = np.random.RandomState(8004)
    mask = np.zeros(batch.inputs_.shape, dtype=bool)
    batch_size = mask.shape[0]
    noise_amount = 0.3
    nr_joints = len(Skeleton.ALL_JOINTS) - 1  # minus root
    nr_frames = mask.shape[2]

    for i in range(batch_size):
        for f in range(nr_frames):
            suppress_joint = rng.binomial(1, noise_amount, nr_joints)
            idxs = np.where(suppress_joint != 0)
            idxs_j = (idxs[0] + 1)*3
            mask[i, idxs_j + 0, f] = True
            mask[i, idxs_j + 1, f] = True
            mask[i, idxs_j + 2, f] = True

    batch_noise = batch.copy()
    batch_noise.inputs_[mask] = 0.0
    return batch_noise, mask


def add_gaussian_noise(batch, std=0.5):
    rng = np.random.RandomState(8004)
    noise = rng.randn(*batch.inputs_.shape)*std
    # add the noise to the velocities
    n_coords = len(Skeleton.ALL_JOINTS)*3
    noise[:, n_coords:n_coords+3] = 0.0
    batch.inputs_ = batch.inputs_ + noise
    return batch


def main(argv):
    if len(argv) > 1:
        # we have some unparsed flags
        raise ValueError('unknown flags: {}'.format(' '.join(argv[1:])))

    # parse the --model flag
    model_name, run_id, discard_foot_contacts, replace_traj = utils.extract_model_name_and_run(FLAGS.model)

    # get the location of the model
    model_path = get_model_path(model_name, run_id)

    # get a random validation batch
    normalizer = get_normalizer(model_path)
    rng = np.random.RandomState(4313)
    feeder = get_feeder(normalizer=None, discard_foot_contacts=discard_foot_contacts, batch_size=FLAGS.samples, rng=rng)
    batch = feeder.random_valid_batch(rng)

    use_gaussian_noise = True
    if use_gaussian_noise:
        # apply gaussian noise to the original input
        batch_noise = add_gaussian_noise(batch, std=1.0)
        ori = np.copy(batch_noise.inputs_)

        # then normalize
        batch_noise.inputs_ = normalizer.normalize(batch_noise.inputs_)

        # show_np_arrays(batch_noise.inputs_[0:1], '', [''])
    else:
        # set values to 0 in both original and normalized version so that we can visualize
        batch.inputs_ = normalizer.normalize(batch.inputs_)
        batch_noise, noise_mask = add_random_zeros(batch)

        # apply perturbation to the original
        ori = np.copy(normalizer.unnormalize(batch.inputs_))
        ori[noise_mask] = 0.0

        # get matrix showing which joints were masked out
        # joints_mask = noise_mask[:, np.array(list(range(1, 22)))*3]
        # ids = [id_ for (_, _, id_, _) in batch.all_entries()]
        # for i in range(joints_mask.shape[0]):
        #     m = joints_mask[i]
        #     fname = 'zero_denoising_mask_id{}.txt'.format(ids[i])
        #     np.savetxt(fname, np.array(m, dtype=np.int64), delimiter=',', fmt='%d')

    # get the prediction
    pred = get_prediction_from_model(model_path, test_batch=batch_noise)

    # unnormalize the prediction
    pred_un = normalizer.unnormalize(pred)
    targets = batch_noise.targets
    # targets = normalizer.unnormalize(batch_noise.targets)

    # Compute joint reconstruction error.
    from tbase.skeleton import to_global_batched
    body_dim = len(Skeleton.ALL_JOINTS) * 3
    traj = targets[:, body_dim:body_dim + 3]
    pred_global = to_global_batched(pred_un, override_trajectory=traj,
                                    override_root=targets[:, 0:3])  # (N, n_joints, 3, seq_len)
    ori_global = to_global_batched(ori, override_trajectory=traj,
                                   override_root=targets[:, 0:3])
    targets_global = to_global_batched(targets)  # (N, n_joints, 3, seq_len)

    # Compute 3D reconstruction error per joint and frame w.r.t to target.
    joint_error = np.linalg.norm(pred_global - targets_global, axis=2)
    joint_error = np.transpose(joint_error, [0, 2, 1])  # (N, seq_len, n_joints)
    print('3d joint error:', np.mean(joint_error), np.std(joint_error))

    # Compute initial reconstruction error between noisy and target sample.
    joint_error_init = np.linalg.norm(ori_global - targets_global, axis=2)
    joint_error_init = np.transpose(joint_error_init, [0, 2, 1])  # (N, seq_len, n_joints)
    print('3d joint error init:', np.mean(joint_error_init), np.std(joint_error_init))

    base_name = 'gaussian_denoising' if use_gaussian_noise else 'zero_denoising'
    ori_names = ['{}_ori_id{}'.format(base_name, id_) for (_, _, id_, _) in batch.all_entries()]
    pred_names = ['{}_pred_id{}'.format(base_name, id_) for (_, _, id_, _) in batch.all_entries()]
    names = [ori_names, pred_names]

    # visualize
    show_motions([ori, pred_un], interp_idxs=np.array([0]*pred_un.shape[0]),
                 names=names, draw_cylinders=True)


if __name__ == '__main__':
    tf.app.run()

