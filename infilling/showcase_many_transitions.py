"""
Showcase several transitions with model that supports variable sequence lengths.
"""
from showcase_utils import *
from plotter import show_np_arrays
from models import compute_bone_lengths_np

fs = tf.app.flags
fs.DEFINE_string('model', 'none',
                 'which model to use for the prediction, expected format is "model_name/run_id[ft]" [none]')

FLAGS = fs.FLAGS


def main(argv):
    if len(argv) > 1:
        # we have some unparsed flags
        raise ValueError('unknown flags: {}'.format(' '.join(argv[1:])))

    # parse the --model flag
    model_name, run_id, discard_foot_contacts, replace_traj = utils.extract_model_name_and_run(FLAGS.model)

    # get the location of the model
    model_path = get_model_path(model_name, run_id)

    # selecte the key sequences
    walk = KeySequence(642, 0, 50)
    walk2 = KeySequence(1447, 0, 45)
    stand_turn = KeySequence(1485, 80, 120)
    run = KeySequence(1961, 146, 186)
    jump_one_legged = KeySequence(1895, 78, 83)
    bow = KeySequence(2307, 85, 140)
    jump = KeySequence(819, 151, 161)
    sitting = KeySequence(1789, 148, 238)
    stand_old = KeySequence(533, 110, 150)
    balet = KeySequence(1871, 113, 160)
    squat = KeySequence(1998, 34, 44)
    kick = KeySequence(1753, 175, 190)
    punch = KeySequence(603, 105, 130)
    punch2 = KeySequence(2512, 78, 93)
    crouch = KeySequence(174, 230, 239)
    jumping_jack = KeySequence(2124, 140, 155)
    skateboard = KeySequence(492, 188, 209)
    backwards_walk = KeySequence(1517, 39, 71)
    backwards_walk2 = KeySequence(857, 168, 213)
    handstand = KeySequence(1731, 173, 190)
    lying = KeySequence(2146, 30, 75)
    ladder_down = KeySequence(793, 122, 146)
    stairs_up = KeySequence(200, 133, 163)
    big_step = KeySequence(280, 107, 151)

    # awesome sequence
    clips = [walk, sitting, run, jump, walk, jump_one_legged, walk, stand_old, bow, stand_turn, balet, walk, squat,
             walk, kick, punch]
    gaps = [100, 100, 100, 100, 50, 100, 100, 100, 100, 100, 50, 50, 50, 100, 100]
    name = 'awesome_transition'

    # compute some stats
    print('Nr of key sequences: {}'.format(len(clips)))
    sizes = np.array([c.context for c in clips])
    print('Average key sequence length: {}'.format(np.average((sizes))))
    print('Min key sequence length: {}'.format(np.amin((sizes))))
    print('Max key sequence length: {}'.format(np.amax((sizes))))

    test_clip, interp_idxs = create_test_clip_from_key_frames(clips, gaps, model_path)

    # visualize input
    # mask = np.zeros(test_clip.shape, dtype=np.bool)
    # mask[:, np.where((test_clip == 0.0).all(axis=0))[0]] = True
    # test_clip_v = np.ma.array(test_clip, mask=mask)
    # show_np_arrays([test_clip_v[:, 230:980]], 'many transitions', [''])

    # get the prediction
    pred = get_prediction_from_model(model_path, test_clip=test_clip)

    # unnormalize the prediction
    normalizer = get_normalizer(model_path)
    pred_un = normalizer.unnormalize(pred)[0]

    # visualize
    show_one_motion([pred_un], interp_idxs, names=[name])


if __name__ == '__main__':
    tf.app.run()
