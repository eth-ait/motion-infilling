"""
Loads the BVH files that make up the databases and processes them into the format required by our training algorithm.
It does NOT subdivide the clips into overlapping windows and does NOT split the data set into training and validation.
For this, use the script `extract_data_splits.py`.

This code is mostly copied from Holden et al. and tweaked to our purposes where necessary.
"""
import os
import numpy as np
import scipy.ndimage.filters as filters

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def process_file(filename, window=240, window_step=120):
    anim, names, frametime = BVH.load(filename)

    """ Convert to 60 fps """
    anim = anim[::2]

    """ Do FK """
    global_positions = Animation.positions_global(anim)

    """ Remove Uneeded Joints """
    positions = global_positions[:, np.array([
        0,
        2, 3, 4, 5,
        7, 8, 9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]

    """ Put on Floor """
    # positions is (seq_length, n_joints, 3)
    fid_l, fid_r = np.array([4, 5]), np.array([8, 9])
    foot_heights = np.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    positions[:, :, 1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:, 0] * np.array([1, 0, 1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
    positions = np.concatenate([reference[:, np.newaxis], positions], axis=1)

    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

    """ Get Root Velocity """
    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()

    """ Remove Translation """
    positions[:, :, 0] = positions[:, :, 0] - positions[:, 0:1, 0]
    positions[:, :, 2] = positions[:, :, 2] - positions[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
    across1 = positions[:, hip_l] - positions[:, hip_r]
    across0 = positions[:, sdr_l] - positions[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    positions = rotation * positions

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

    return positions


def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']


def export_db(input_path, output_path):
    print('\nprocessing db {} ...'.format(input_path.split('/')[-1]))
    all_files = get_files(input_path)
    all_clips = []
    lengths = []
    for i, item in enumerate(all_files):
        print('\r\tprocessing {} of {} ({})'.format(i, len(all_files), item), end='')
        clips = process_file(item)
        all_clips.append(clips)
        lengths.append(clips.shape[0])
    data_clips = np.array(all_clips)

    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    min_length = np.amin(lengths)
    max_length = np.amax(lengths)
    print('\ngathered {} clips of mean length {} (+/- {}) max length {} min length {}'.format(
        data_clips.shape[0], mean_length, std_length, max_length, min_length))

    np.savez_compressed(output_path, clips=data_clips)


if __name__ == '__main__':
    data_base_path = '/path_to_data_from_holden/motionsynth_data/data/processed/'
    output_path = '../data_preprocessed/raw/'

    dbs = [(os.path.join(data_base_path, 'cmu'), os.path.join(output_path, 'data_cmu.npz')),
           (os.path.join(data_base_path, 'hdm05'), os.path.join(output_path, 'data_hdm05.npz')),
           (os.path.join(data_base_path, 'edin_locomotion'), os.path.join(output_path, 'data_edin_locomotion.npz')),
           (os.path.join(data_base_path, 'edin_xsens'), os.path.join(output_path, 'data_edin_xsens.npz')),
           (os.path.join(data_base_path, 'edin_kinect'), os.path.join(output_path, 'data_edin_kinect.npz')),
           (os.path.join(data_base_path, 'edin_misc'), os.path.join(output_path, 'data_edin_misc.npz')),
           (os.path.join(data_base_path, 'mhad'), os.path.join(output_path, 'data_mhad.npz')),
           (os.path.join(data_base_path, 'edin_punching'), os.path.join(output_path, 'data_edin_punching.npz')),
           (os.path.join(data_base_path, 'edin_terrain'), os.path.join(output_path, 'data_edin_terrain.npz'))]

    for (db_path, out_path) in dbs:
        export_db(db_path, out_path)

