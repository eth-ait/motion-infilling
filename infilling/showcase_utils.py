import os
import time

import numpy as np
import tensorflow as tf
from tbase import utils

from tbase.data_loader import LoaderV2, FeederV2, Batch
from tbase.normalizers import Normalizer
from tbase.skeleton import SkeletonSequence
from tbase.visualizer import Visualizer

import flags_parser
from models import VanillaAutoencoder
from models import compute_bone_lengths_np
import matplotlib.pyplot as plt


def get_feeder(normalizer, discard_foot_contacts, batch_size=20, rng=np.random):
    data_path = flags_parser.get_data_path()
    loader = LoaderV2(train_path=os.path.join(data_path, 'train'),
                      valid_path=os.path.join(data_path, 'valid'),
                      normalizer=None,  # done manually so that function can be used with and without normalizer
                      discard_foot_contacts=discard_foot_contacts)
    data = loader.get_validation_unnormalized_all()
    data = normalizer.normalize(data) if normalizer is not None else data
    feeder = FeederV2(data_train=None,
                      data_valid=data,
                      batch_size=batch_size,
                      rng=rng)
    return feeder


def get_clips_from_disk(filenames, normalizer):
    clips = []
    for f_name in filenames:
        data = np.load(f_name)['data']
        data = normalizer.normalize(data)
        clips.append(data[0])
    return clips


def get_model_path(model_name, run_id):
    return flags_parser.create_model_path(
        flags_parser.get_checkpoints_path(), model_name, run_id)


def get_normalizer(model_path):
    normalizer = Normalizer.from_disk(model_path)
    normalizer.load(model_path)
    return normalizer


def clips_to_batch(clips):
    # package the clips into a dummy batch
    input_ = np.expand_dims(clips, 0) if len(clips.shape) == 2 else clips
    batch = Batch(input_=input_,
                  targets=input_,
                  ids=np.zeros(input_.shape[0]))
    return batch


def get_prediction_from_model(model_path, test_clip=None, test_batch=None):
    assert (test_clip is not None and test_batch is None) or (test_clip is None and test_batch is not None)

    tf.reset_default_graph()
    latest_checkpoint = utils.get_latest_checkpoint(model_path)
    if latest_checkpoint is None:
        raise RuntimeError('could not find checkpoint {}'.format(model_path))

    print('restoring checkpoint {} ...'.format(latest_checkpoint))
    model = VanillaAutoencoder.build_from_metagraph(checkpoint=latest_checkpoint,
                                                    perturbator=None,
                                                    use_masked_loss=False)

    # compute reconstruction, only use fraction of available GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # restore the model
        model.load(sess, latest_checkpoint)

        # create batch
        if test_clip is not None:
            batch = clips_to_batch(test_clip)
        else:
            batch = test_batch

        tick = time.time()
        # get the prediction
        [recn] = model.evaluate_batch(sess, batch, custom_fetch=[model.reconstructed])
        print('inference time: {} secs'.format((time.time() - tick)))

    return recn


def create_test_clip_from_key_frames(key_sequences, gap_lengths, model_path):
    assert len(gap_lengths) == len(key_sequences) - 1

    # load all the sequences
    normalizer = get_normalizer(model_path)
    feeder = get_feeder(normalizer, True)
    [c.load(feeder) for c in key_sequences]

    # stitch together the final test clip
    test_clip = key_sequences[0].frames
    n_dim = test_clip.shape[0]
    interp_idxs = []
    for i in range(0, len(key_sequences)-1):
        interp_idxs += list(range(test_clip.shape[1]-1, test_clip.shape[1]-1 + gap_lengths[i]))
        test_clip = np.concatenate([test_clip, np.zeros([n_dim, gap_lengths[i]]), key_sequences[i + 1].frames], axis=1)

    print('test clip shape:')
    print(test_clip.shape)
    print('gap size {} ({:.2f}%)'.format(np.sum(gap_lengths), np.sum(gap_lengths)/test_clip.shape[1]*100.0))

    return test_clip, interp_idxs


def show_one_motion(reconstruction, interp_idxs, draw_cylinders=True, names=None, static_points=None):
    """
    Show one motion clip and possibly compare several outputs. I.e. `reconstruction` is a list of (dim, seq_length)
    arrays. 
    """
    vi = Visualizer(draw_cylinders=draw_cylinders)
    offset = 25.0
    sks = [SkeletonSequence(sequence=r,
                            name=names[i] if names is not None else str(i),
                            interp=interp_idxs,
                            x_offset=i*offset) for i, r in enumerate(reconstruction)]
    if static_points is not None:
        [sks[i].set_static_points(ps) for i, ps in enumerate(static_points)]

    # bone_lengths = compute_bone_lengths_np(sks[0]._joints.reshape((1, 66, -1)) / 0.16).squeeze()
    # bones = [2, 10]
    # bone_names = ['right lower leg', 'spine']
    # for ii, bn in zip(bones, bone_names):
    #     plt.plot(bone_lengths[ii], label=bn)
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('Bone Length [cm]')
    # plt.grid()
    # plt.show()

    vi.load_skeleton_sequences(sks)
    vi.show_()


def show_motions(reconstructions, interp_idxs, names=None, draw_cylinders=True, static_points=None):
    """
    Show several motion clips one after the other, where each clip can be compared to several outpus. I.e. 
    `reconstructions` is a list of (n, dim, seq_length) arrays.
    """
    batch_size = reconstructions[0].shape[0]
    for i in range(0, batch_size):
        reconstruction = [r[i] for r in reconstructions]
        ns = [n[i] for n in names]
        sp = [s[i] for s in static_points] if static_points is not None else None
        show_one_motion(reconstruction, interp_idxs[i], draw_cylinders, names=ns, static_points=sp)


class KeySequence(object):
    def __init__(self, id_, start_frame=0, end_frame=240):
        self.id_ = id_
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frames = None

    def load(self, feeder):
        batch = feeder.valid_batch_from_idxs(np.array([self.id_]))
        data = batch[0].inputs_[0]
        self.frames = data[:, self.start_frame:self.end_frame]

    @property
    def context(self):
        return self.end_frame - self.start_frame

