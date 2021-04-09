"""
Mask one or several joints fully and reconstruct.
"""
from showcase_utils import *
from tbase.skeleton import Skeleton

fs = tf.app.flags
fs.DEFINE_string('model', 'none', 'which model to use for the prediction, expected format is "model_name/run_id[ft]" [none]')
fs.DEFINE_integer('samples', 20, 'amount of examples to be shown')

FLAGS = fs.FLAGS


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
    feeder = get_feeder(normalizer, discard_foot_contacts=True, batch_size=FLAGS.samples)
    batch = feeder.valid_batch_from_idxs([1214, 1690, 806, 664])

    # choose which joints to mask
    joint_id = [Skeleton.RIGHT_KNEE, Skeleton.RIGHT_TOE, Skeleton.LEFT_KNEE]

    # do the masking
    batch_masked = batch.copy()
    mask = np.zeros(batch.inputs_.shape, dtype=np.int64)
    for i, (ori, _, _, _) in enumerate(batch_masked.all_entries()):
        for joint in joint_id:
            ori[joint*3:joint*3+3] = 0.0
            mask[i, joint*3:joint*3+3] = 1

    # get the prediction
    pred = get_prediction_from_model(model_path, test_batch=batch_masked)

    # unnormalize the prediction
    pred_un = normalizer.unnormalize(pred)

    # original
    ori = normalizer.unnormalize(batch.inputs_)

    base_name = 'joint'
    ori_names = ['{}_ori_id{}'.format(base_name, id_) for (_, _, id_, _) in batch.all_entries()]
    pred_names = ['{}_pred_id{}'.format(base_name, id_) for (_, _, id_, _) in batch.all_entries()]
    names = [ori_names, pred_names]

    # dump all the masks
    ids = [id_ for (_, _, id_, _) in batch.all_entries()]
    for i in range(mask.shape[0]):
        m = mask[i]
        m = m[np.array(list(range(1, 22))) * 3, :]
        fname = '{}_mask_id{}.txt'.format(base_name, ids[i])
        np.savetxt(fname, np.array(m, dtype=np.int64), delimiter=',', fmt='%d')

    # visualize
    show_motions([ori, pred_un], interp_idxs=np.array([0]*pred_un.shape[0]), names=names)


if __name__ == '__main__':
    tf.app.run()

