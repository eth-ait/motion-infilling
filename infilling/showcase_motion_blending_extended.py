"""
Fix the hand in a gap during walking motion.
"""
from showcase_utils import *
from tbase.skeleton import Skeleton

fs = tf.app.flags
fs.DEFINE_string('model', 'none',
                 'which model to use for the prediction, expected format is "model_name/run_id[ft]" [none]')
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

    # get a walking and a running clip
    walk = KeySequence(642, 0, 50)
    run = KeySequence(1961, 146, 186)
    walk2 = KeySequence(1447, 0, 45)

    # create a gap between them
    clips = [walk, run, walk2]
    gaps = [100, 120]
    name = 'motion_blending_zoo'
    test_clip, interp_idxs = create_test_clip_from_key_frames(clips, gaps, model_path)
    start_gap1 = clips[0].context
    end_gap1 = start_gap1 + gaps[0]
    start_gap2 = end_gap1 + clips[1].context
    end_gap2 = start_gap2 + gaps[1]

    # get the grabing motion
    clip_grab = feeder.valid_batch_from_idxs([2225]).inputs_[0]
    effectors_g = [Skeleton.RIGHT_HAND, Skeleton.RIGHT_WRIST, Skeleton.RIGHT_ELBOW]
    effector_idxs_g = np.reshape(np.array([list(range(e * 3, e * 3 + 3)) for e in effectors_g]), [-1])
    start_effector_g = 8
    end_effector_g = 23
    end_effector_g_data = clip_grab[effector_idxs_g, start_effector_g:end_effector_g]

    # get the kicking motion
    clip_kick = feeder.valid_batch_from_idxs([1753]).inputs_[0]
    effectors_k = [Skeleton.LEFT_KNEE, Skeleton.LEFT_HEEL, Skeleton.LEFT_TOE]
    effector_idxs_k = np.reshape(np.array([list(range(e * 3, e * 3 + 3)) for e in effectors_k]), [-1])
    start_effector_k = 183
    end_effector_k = 198
    end_effector_k_data = clip_kick[effector_idxs_k, start_effector_k:end_effector_k]

    # insert grabing motion into first gap
    location = (end_gap1 - start_gap1) // 2 + start_gap1
    start_insert_g = location - end_effector_g_data.shape[1] // 2
    end_insert_g = start_insert_g + end_effector_g_data.shape[1]
    test_clip[effector_idxs_g, start_insert_g:end_insert_g] = end_effector_g_data

    # insert the kicking motion into the second gap
    location = (end_gap2 - start_gap2) // 2 + start_gap2
    start_insert_k = location - end_effector_k_data.shape[1] // 2
    end_insert_k = start_insert_k + end_effector_k_data.shape[1]
    test_clip[effector_idxs_k, start_insert_k:end_insert_k] = end_effector_k_data

    # also insert the punching motion in the second gap
    location = (end_gap2 - start_gap2) // 2 + start_gap2
    start_insert_p = location - end_effector_g_data.shape[1] // 2
    end_insert_p = start_insert_p + end_effector_g_data.shape[1]
    test_clip[effector_idxs_g, start_insert_p:end_insert_p] = end_effector_g_data

    # get the prediction
    pred = get_prediction_from_model(model_path, test_clip=test_clip)
    pred_un = normalizer.unnormalize(pred)

    # create a mask that shows which joints were given as end effectors for which frames and dump it
    mask = np.zeros(test_clip.shape, dtype=np.int64)
    mask[effector_idxs_g, start_insert_g:end_insert_g] = 1
    mask[effector_idxs_k, start_insert_k:end_insert_k] = 1
    mask[effector_idxs_g, start_insert_p:end_insert_p] = 1
    mask = mask[np.array(list(range(1, 22))) * 3, :]
    np.savetxt('{}_mask.txt'.format(name), mask, delimiter=',', fmt='%d')

    # visualize
    interp_idxs = [np.concatenate(
        [np.arange(start_gap1, end_gap1), np.arange(start_gap2, end_gap2)])] * pred_un.shape[0]
    show_motions([pred_un], interp_idxs=np.array(interp_idxs),
                 names=[[name]], draw_cylinders=True)


if __name__ == '__main__':
    tf.app.run()
