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

    # get a walking and a running clip
    walk = KeySequence(642, 0, 50)
    run = KeySequence(1961, 146, 186)

    # create a gap between them
    clips = [walk, run]
    gaps = [100]
    name = 'ui_showcase_with_blending'
    test_clip, interp_idxs = create_test_clip_from_key_frames(clips, gaps, model_path)
    start_gap = walk.context
    end_gap = start_gap + gaps[0]

    # insert the grabing motion in the middle of this gap
    clip_grab = feeder.valid_batch_from_idxs([2225]).inputs_[0]
    effectors = [Skeleton.RIGHT_HAND, Skeleton.RIGHT_WRIST, Skeleton.RIGHT_ELBOW]
    effector_idxs = np.reshape(np.array([list(range(e*3, e*3+3)) for e in effectors]), [-1])
    start_effector = 8
    end_effector = 23
    end_effector_data = clip_grab[effector_idxs, start_effector:end_effector]

    # insert end effector into sequence
    location = test_clip.shape[1] // 2
    start_insert = location - end_effector_data.shape[1]//2
    end_insert = start_insert + end_effector_data.shape[1]
    test_clip[effector_idxs, start_insert:end_insert] = end_effector_data

    # get the prediction
    pred = get_prediction_from_model(model_path, test_clip=test_clip)
    pred_un = normalizer.unnormalize(pred)

    # create a mask that shows which joints were given as end effectors for which frames and dump it
    mask = np.zeros(test_clip.shape, dtype=np.int64)
    mask[effector_idxs, start_insert:end_insert] = 1
    mask = mask[np.array(list(range(1, 22))) * 3, :]
    np.savetxt('{}_mask.txt'.format(name), mask, delimiter=',', fmt='%d')

    # visualize
    show_motions([pred_un], interp_idxs=np.array([np.arange(start_gap, end_gap)]*pred_un.shape[0]),
                 names=[[name]], draw_cylinders=True)


if __name__ == '__main__':
    tf.app.run()

