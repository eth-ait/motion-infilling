"""
Module to interface with data loader.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from tbase.skeleton import Skeleton


def to_3_channels_rep(data):
    """
    Converts the input in shape (batch_size, 73, width) to (batch_size, 22 + 7, width, 3). Can also handle missing
    foot contacts, in which case the input shape is (batch_size, 69, width)
    """
    assert data.shape[1] >= 69 and (len(data.shape) == 3 or len(data.shape) == 4)


    seq_length = data.shape[2]
    batch_size = data.shape[0]
    n_joints = len(Skeleton.ALL_JOINTS)

    data_c = np.copy(data)
    pose = data_c[:, :n_joints*3, ...]
    rest = data_c[:, n_joints*3:, ...]

    # convert pose from (batch_size, 66, seq_length) to (batch_size, 22, seq_length, 3)
    pose_r = np.reshape(pose, [batch_size, n_joints, 3, seq_length])
    pose_3 = np.transpose(pose_r, [0, 1, 3, 2])

    # zero pad channels for remaining data
    zeros = np.zeros(rest.shape + (2,))
    rest_concat = np.concatenate([np.expand_dims(rest, -1), zeros], axis=3)

    # paste back together
    converted = np.concatenate([pose_3, rest_concat], axis=1)
    assert converted.shape[3] == 3 and converted.shape[2] == seq_length and converted.shape[0] == batch_size
    assert converted.shape[1] == n_joints + rest.shape[1]
    return converted


def to_1_channel_rep(data):
    """
    The inverse of `to_3_channels_rep`.
    """
    if len(data.shape) == 3 and data.shape[1] >= 69:
        # this is already the required format
        return data

    if len(data.shape) == 4 and data.shape[3] == 1:
        # this has only one channel so just remove that
        return np.squeeze(data, -1)

    seq_length = data.shape[2]
    batch_size = data.shape[0]
    n_joints = len(Skeleton.ALL_JOINTS)
    assert len(data.shape) == 4 and data.shape[3] == 3 and (data.shape[1] == n_joints + 7 or data.shape[1] == n_joints + 3)

    data_c = np.copy(data)
    pose = data_c[:, :n_joints, ...]
    rest = data_c[:, n_joints:, ...]

    # convert pose from (batch_size, 22, seq_length, 3) to (batch_size, 66, seq_length)
    pose_r = np.transpose(pose, [0, 1, 3, 2])
    pose_1 = np.reshape(pose_r, [batch_size, n_joints*3, seq_length])

    # get rest of the data, i.e. drop unused channels
    rest_1 = rest[:, :, :, 0]

    # paste back together
    converted = np.concatenate([pose_1, rest_1], axis=1)
    assert converted.shape[0] == batch_size and converted.shape[1] == 66 + rest.shape[1] and converted.shape[2] == seq_length
    return converted


def remove_foot_contacts(data):
    assert data.shape[1] == 73
    return np.delete(data, obj=list(range(data.shape[1] - 4, data.shape[1])), axis=1)


class Databases(object):
    """
    Helper class to define names of available motion capture databases.
    """
    CMU = 'data_cmu.npz'
    HDM05 = 'data_hdm05.npz'
    MHAD = 'data_mhad.npz'
    EDIN_LOCOMOTION = 'data_edin_locomotion.npz'
    EDIN_XSENS = 'data_edin_xsens.npz'
    EDIN_MISC = 'data_edin_misc.npz'
    EDIN_PUNCHING = 'data_edin_punching.npz'

    # helper dict to retrieve database files also from strings
    MAPPING = {'cmu': CMU, 'hdm05': HDM05, 'mhad': MHAD, 'edin_locomotion': EDIN_LOCOMOTION,
               'edin_xsens': EDIN_XSENS, 'edin_misc': EDIN_MISC, 'edin_punching': EDIN_PUNCHING}

    @classmethod
    def from_str(cls, name):
        """Retrieves the name of the DB file for a given string."""
        n = name.lower()
        return cls.MAPPING[n]


class LoaderV2(object):
    """
    Loader that can handle data we preprocessed ourselves, i.e. not using Holden data directly.
    """
    def __init__(self, train_path, valid_path, normalizer=None, discard_foot_contacts=False):
        self.train_path = train_path
        self.valid_path = valid_path
        self.splits_path = [self.train_path, self.valid_path]
        self.normalizer = normalizer
        self.discard_foot_contacts = discard_foot_contacts
        self.all_dbs = [Databases.CMU, Databases.HDM05, Databases.MHAD, Databases.EDIN_LOCOMOTION,
                        Databases.EDIN_XSENS, Databases.EDIN_MISC, Databases.EDIN_PUNCHING]

    def _get_split_unnormalized(self, db_name, split):
        data = np.load(os.path.join(self.splits_path[split], db_name))['clips']
        data = np.swapaxes(data, 1, 2)
        data = self.remove_foot_contact_info(data)
        return data

    def get_training_unnormalized(self, db_name):
        return self._get_split_unnormalized(db_name, split=0)

    def get_validation_unnormalized(self, db_name):
        return self._get_split_unnormalized(db_name, split=1)

    def get_data_unnormalized(self, db_name):
        return self.get_training_unnormalized(db_name), self.get_validation_unnormalized(db_name)

    def get_validation_unnormalized_all(self):
        data = []
        for db in self.all_dbs:
            data.append(self.get_validation_unnormalized(db))
        return np.concatenate(data, axis=0)

    def get_training_unnormalized_all(self):
        data = []
        for db in self.all_dbs:
            data.append(self.get_training_unnormalized(db))
        return np.concatenate(data, axis=0)

    def get_data_unnormalized_all(self):
        return self.get_training_unnormalized_all(), self.get_validation_unnormalized_all()

    def load_training_all(self):
        data = self.get_training_unnormalized_all()
        return self.normalizer.normalize(data)

    def load_validation_all(self):
        data = self.get_validation_unnormalized_all()
        return self.normalizer.normalize(data)

    def load_all(self):
        # Note that it is important that the training data is loaded first, because this function internally computes
        # the normalization statistics which are then reused for all subsequent calls.
        data_train_n = self.load_training_all()
        data_valid_n = self.load_validation_all()
        return data_train_n, data_valid_n

    def remove_foot_contact_info(self, data):
        if self.discard_foot_contacts:
            return remove_foot_contacts(data)
        else:
            return data


class Batch(object):
    """
    Represents one minibatch.
    """
    def __init__(self, input_, targets, ids, mask=None):
        self.inputs_ = input_
        self.targets = targets
        self.ids = ids
        self.mask = mask if mask is not None else np.zeros(input_.shape)
        self.batch_size = self.inputs_.shape[0]

    def __getitem__(self, item):
        if not 0 <= item < self.batch_size:
            raise IndexError('batch index {} out of bounds for batch size {}'.format(item, self.batch_size))
        return Batch(self.inputs_[item:item+1, ...],
                     self.targets[item:item+1, ...],
                     self.ids[item:item+1, ...],
                     self.mask[item:item+1, ...])

    def all_entries(self):
        for i in range(self.batch_size):
            yield self.inputs_[i], self.targets[i], self.ids[i], self.mask[i]

    def perturbate(self, perturbator, reapply=False):
        if perturbator is None:
            self.targets = self.inputs_
        else:
            if reapply:
                perturbated, mask = perturbator.reapply_last_perturbation(self.inputs_)
            else:
                perturbated, mask = perturbator.perturbate(self.inputs_)
            self.targets = self.inputs_
            self.inputs_ = perturbated
            self.mask = mask

    def remove_foot_contacts(self):
        self.inputs_ = remove_foot_contacts(self.inputs_)
        self.targets =remove_foot_contacts(self.targets)
        self.mask = remove_foot_contacts(self.mask)

    def copy(self):
        return Batch(np.copy(self.inputs_),
                     np.copy(self.targets),
                     np.copy(self.ids),
                     np.copy(self.mask))


class AbstractFeeder(object):
    def _get_batch(self, split, batch_ptr):
        """
        Get the specified batch.
        :param split: Which split to access.
        :param batch_ptr: Which batch to access, i.e. index between 0 and number of batches for the given split.
        :return: The retrieved batch.
        """
        raise NotImplementedError('Method is abstract.')

    def _next_batch_from_split(self, split):
        """
        Returns the next available batch for the chosen split. Circular access if overflow happens.
        :param split: 0, 1, or 2 for 'train', 'valid', or 'test' respectively.
        :return: The next available batch
        """
        raise NotImplementedError('Method is abstract.')

    def _all_split_batches(self, split):
        """
        Generator function looping over all available batches in the given split.
        """
        for i in range(self._n_batches_split(split)):
            yield self._next_batch_from_split(split)

    def _n_batches_split(self, split):
        raise NotImplementedError('Method is abstract.')

    def _random_batch_from_split(self, split, rng=np.random):
        """
        Returns a random batch from the requested split.
        """
        batch_ptr = rng.randint(0, self._n_batches_split(split))
        batch = self._get_batch(split, batch_ptr)
        return batch

    def reshuffle_train(self, rng=np.random):
        """
        Reshuffles the training data set.
        """
        raise NotImplementedError('Method is abstract.')

    def n_batches_train(self):
        return self._n_batches_split(0)

    def n_batches_valid(self):
        return self._n_batches_split(1)

    def n_batches_test(self):
        return self._n_batches_split(2)

    def next_batch_train(self):
        return self._next_batch_from_split(0)

    def next_batch_valid(self):
        return self._next_batch_from_split(1)

    def next_batch_test(self):
        return self._next_batch_from_split(2)

    def train_batches(self):
        return self._all_split_batches(0)

    def valid_batches(self):
        return self._all_split_batches(1)

    def test_batches(self):
        return self._all_split_batches(2)

    def random_train_batch(self, rng):
        return self._random_batch_from_split(0, rng)

    def random_valid_batch(self, rng):
        return self._random_batch_from_split(1, rng)

    def random_test_batch(self, rng):
        return self._random_batch_from_split(2, rng)

    def valid_batch_from_idxs(self, indices):
        raise NotImplementedError('Method is abstract.')


class FeederV2(AbstractFeeder):
    def __init__(self, data_train, data_valid, batch_size, rng=np.random):
        # `data_train` can be None. If so, only the "validation" functions can be used.
        self._data_train = data_train
        self._data_valid = data_valid
        self._batch_size = batch_size
        self._splits = [data_train, data_valid]
        self._rng = rng

        # compute how many batches per split
        n_train = int(np.ceil(float(len(data_train)) / float(batch_size))) if data_train is not None else 0
        n_valid = int(np.ceil(float(len(data_valid)) / float(batch_size)))

        print('Number of batches per split: train {} valid {}'.format(
            n_train, n_valid))

        self._n_batches = [n_train, n_valid]  # number of batches per split
        self._batch_ptrs = [0, 0]  # pointers to the next available batch per split
        self._indices = [np.arange(0, len(data_train) if data_train is not None else 0), np.arange(0, len(data_valid))]

        # reshuffle the indices
        self._rng.shuffle(self._indices[0])
        self._rng.shuffle(self._indices[1])

    def _get_batch_ptr(self, split):
        return self._batch_ptrs[split]

    def _set_batch_ptr(self, split, value):
        new_val = value if value < self._n_batches[split] else 0
        self._batch_ptrs[split] = new_val

    def _get_batch(self, split, batch_ptr):
        """
        Get the specified batch.
        :param split: Which split to access.
        :param batch_ptr: Which batch to access, i.e. index between 0 and number of batches for the given split.
        :return: The retrieved batch.
        """
        assert 0 <= batch_ptr < self._n_batches_split(split), 'batch pointer out of range'

        start_idx = batch_ptr * self._batch_size
        end_idx = (batch_ptr + 1) * self._batch_size

        # because we want to use all available data, must be careful that `end_idx` is valid
        end_idx = end_idx if end_idx <= len(self._indices[split]) else len(self._indices[split])
        indices = self._indices[split][start_idx:end_idx]
        inputs = self._splits[split][indices, ...]
        targets = np.copy(inputs)  # at the moment targets are inputs
        batch = Batch(inputs, targets, indices)
        return batch

    def _next_batch_from_split(self, split):
        """
        Returns the next available batch for the chosen split. Circular access if overflow happens.
        :param split: 0, 1, or 2 for 'train', 'valid', or 'test' respectively.
        :return: The next available batch
        """
        if split > 1:
            return Batch(np.array([]), np.array([]), np.array([]))

        batch_ptr = self._get_batch_ptr(split)
        next_batch = self._get_batch(split, batch_ptr)
        self._set_batch_ptr(split, batch_ptr + 1)
        return next_batch

    def _n_batches_split(self, split):
        return self._n_batches[split] if split <= 1 else 0

    def reshuffle_train(self, rng=np.random):
        """
        Reshuffles the training data set.
        """
        rng.shuffle(self._indices[0])

    def valid_batch_from_idxs(self, indices):
        data = self._splits[1][indices, ...]
        return Batch(data, data, indices)
