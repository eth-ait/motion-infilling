"""
Fix the hand in a gap during walking motion.
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

    # access data
    normalizer = get_normalizer(model_path)
    feeder = get_feeder(normalizer=normalizer, discard_foot_contacts=True, batch_size=FLAGS.samples)

    # get the walking and grabing batch
    clip_walking = feeder.valid_batch_from_idxs([642]).inputs_[0]
    clip_walking_ori = np.copy(clip_walking)

    # mask walking batch somewhere in the middle with fixed gap size
    gap_size = 120
    location = clip_walking.shape[1]//2
    start_gap = location-gap_size//2
    end_gap = start_gap + gap_size
    clip_walking[:, start_gap:end_gap] = 0.0

    do_grab = False
    if do_grab:
        # get end effector position for grabing
        clip_grab = feeder.valid_batch_from_idxs([2225]).inputs_[0]
        effectors = [Skeleton.RIGHT_HAND, Skeleton.RIGHT_WRIST, Skeleton.RIGHT_ELBOW]
        effector_idxs = np.reshape(np.array([list(range(e*3, e*3+3)) for e in effectors]), [-1])
        start_effector = 8
        end_effector = 23
        end_effector_data = clip_grab[effector_idxs, start_effector:end_effector]
        base_name = 'end_effector_grab'
    else:
        # get an end effector position for kicking
        clip_kick = feeder.valid_batch_from_idxs([1753]).inputs_[0]
        effectors = [Skeleton.LEFT_KNEE, Skeleton.LEFT_HEEL, Skeleton.LEFT_TOE]
        effector_idxs = np.reshape(np.array([list(range(e * 3, e * 3 + 3)) for e in effectors]), [-1])
        start_effector = 183
        end_effector = 198
        end_effector_data = clip_kick[effector_idxs, start_effector:end_effector]
        base_name = 'end_effector_kick'

    # insert end effector into sequence
    start_insert = location - end_effector_data.shape[1]//2
    end_insert = start_insert + end_effector_data.shape[1]
    clip_walking[effector_idxs, start_insert:end_insert] = end_effector_data

    # get the prediction
    pred = get_prediction_from_model(model_path, test_clip=clip_walking)

    # unnormalize the prediction
    pred_un = normalizer.unnormalize(pred)
    ori = normalizer.unnormalize(clip_walking_ori[np.newaxis, ...])

    ori_names = ['{}_original_motion'.format(base_name)]
    pred_names = ['{}_with_end_effector'.format(base_name)]
    names = [ori_names, pred_names]

    # create a mask that shows which joints were given as end effectors for which frames and dump it
    mask = np.zeros(clip_walking.shape, dtype=np.int64)
    mask[effector_idxs, start_insert:end_insert] = 1
    mask = mask[np.array(list(range(1, 22))) * 3, :]
    np.savetxt('{}_mask.txt'.format(base_name), mask, delimiter=',', fmt='%d')

    # visualize
    show_motions([ori, pred_un], interp_idxs=np.array([np.arange(start_gap, end_gap)]*pred_un.shape[0]),
                 names=names, draw_cylinders=True)


if __name__ == '__main__':
    tf.app.run()

