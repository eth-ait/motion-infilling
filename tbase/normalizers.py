import os
import importlib
import numpy as np

from tbase.skeleton import Skeleton


class Normalizer(object):
    """
    Normalizes the data.
    """
    def normalize(self, data, force_recomputation=False):
        """
        Normalizes the data.
        :param data: An np-array containing the data.
        :param force_recomputation: If True, the parameters required to do the normalization are recomputed.
        :return: An np-array containint the normalized data.
        """
        raise NotImplementedError()

    def unnormalize(self, data):
        """
        Reverts the effect of the normalization.
        :param data: An np-array containing the normalized data.
        :return: An np-array containing the unnormalized data.
        """
        raise NotImplementedError()

    def save(self, save_path, override_name=None):
        """
        Save necessary parameters to disk.
        :param save_path: Path where parameters will be stored.
        :param override_name: if set, the name written into normalizer.txt will be overwritten with this value
        """
        raise NotImplementedError()

    def load(self, save_path):
        """
        Load the normalizer from a configuration saved via `save()`.
        :param save_path: Path where parameters to be loaded are stored.
        :return: The loaded normalizer object.
        """
        raise NotImplementedError()

    def get_name(self):
        """
        Returns the name of this type of initializer.
        """
        return self.__class__.__name__

    def get_save_location(self, save_path, override_name=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = self.get_name() if override_name is None else override_name
        with open(os.path.join(save_path, 'normalizer.txt'), 'w') as f:
            f.write('{}\n'.format(name))
        return os.path.join(save_path, self.get_name() + '.npz')

    @staticmethod
    def from_disk(save_path):
        """
        Read the file saved in `save()` and return the respective normalizer. This is a default implementation and can
        be changed by subclasses where needed.
        :param save_path: Where the file saved in the call to `save()` is stored.
        :return: The normalizer object.
        """
        fname = os.path.join(save_path, 'normalizer.txt')
        if not os.path.exists(fname):
            raise ValueError('Cannot load normalizer as file "{}" does not exist.'.format(fname))

        # read first line of the file
        with open(fname, 'r') as f:
            class_string = f.readline().strip()

        # instantiate
        if class_string is None:
            raise ValueError('{} is corrupt, could not instantiate normalizer'.format(fname))

        klass = getattr(importlib.import_module(__name__), class_string)
        obj = klass()
        return obj


class IntervalNormalizer(Normalizer):
    """
    Normalizes the data into the interval [low, high]. Note that there is no need for interval normalizers to save
    any parameters (i.e. minimum and maximum of the data to be normalized). It is better to ensure that the normalized
    data falls in the respective range rather than using the same parameters for the normalization across different
    runs/data sets.
    """

    def __init__(self, low, high, x_min=None, x_max=None):
        """
        Constructor. If `x_min` or `x_max` not supplied, they will be calculated from the data passed into
        `normalize()`.

        :param low: Lower boundary of the interval.
        :param high: Upper boundary of the interval.
        :param x_min: Global minimum in the dataset to be normalized.
        :param x_max: Global maximum in the dataset to be normalized.
        """
        self._low = low
        self._high = high
        self._x_min = x_min
        self._x_max = x_max

    def normalize(self, data, force_recomputation=False):
        """
        `data` is expected in the format (nr_data_points, input_dim, window_size). Handles every input_dim seperately,
        i.e. normalization is done seperately for every of those dimensions.
        """
        # find min and max if required
        if self._x_min is None or self._x_max is None or force_recomputation:
            self._x_min = np.amin(data, axis=(0, 2), keepdims=True)
            self._x_max = np.amax(data, axis=(0, 2), keepdims=True)

        # squash data to interval [0, 1], watch out for zero division
        div = (self._x_max - self._x_min)
        div[np.where(div < 1e-6)] = 1.0
        x_std = (data - self._x_min) / div

        # transform to interval [low, high]
        x_norm = x_std * (self._high - self._low) + self._low

        return x_norm

    def unnormalize(self, data):
        if self._x_max is None or self._x_min is None:
            raise ValueError('_x_min and _x_max not set, cannot unnormalize')

        x_std = (data - self._low) / (self._high - self._low)
        x_ori = x_std * (self._x_max - self._x_min) + self._x_min
        return x_ori

    def load(self, save_path):
        """
        Restores internal parameters from `.npz` file found in `save_path`.
        :param save_path: Path to where the `.npz` file for this normalizer is located.
        """
        params = np.load(os.path.join(save_path, self.get_name() + '.npz'))
        self._x_min = params['xmin']
        self._x_max = params['xmax']

    def save(self, save_path, override_name=None):
        """
        Saves the name of the normalizer into a text file named `normalizer.txt`. Also saves the parameters used into
        `save_path` as a `.npz` file.
        :param save_path: Where to save text file and parameters to.
        """
        save_loc = self.get_save_location(save_path, override_name)
        np.savez_compressed(save_loc, xmin=self._x_min, xmax=self._x_max)


class TanhNormalizer(IntervalNormalizer):
    """
    Normalizes data into the interval [-1, 1].
    """
    def __init__(self, x_min=None, x_max=None):
        super().__init__(-1.0, 1.0, x_min, x_max)


class SigmoidNormalizer(IntervalNormalizer):
    """
    Normalizes the data into the interval [0, 1]
    """
    def __init__(self, x_min=None, x_max=None):
        super().__init__(0.0, 1.0, x_min, x_max)


class MeanNormalizer(Normalizer):
    """
    Normalizer that subtracts the mean and divides by the standard deviation.
    """
    def __init__(self, mean=None, std=None):
        self._mean = mean
        self._std = std
        self._feet = Skeleton.FEET_FLATTENED

    def normalize(self, data, force_recomputation=False):
        """
        Subtracts the mean from the data and divides it by the standard deviation. The data is expected in the format
        (nr_data_points, input_dim, window_size) and the normalization is computed for each `input_dim` independently,
        i.e. we compute something like a mean pose.. 

        :param data: an np-array in the format (nr_data_points, input_dim, window_size)
        :param force_recomputation: ignored
        :return: the standardized data
        """
        if self._mean is None:
            self._mean = data.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]

        if self._std is None:
            x_std = np.std(np.std(data, axis=2, keepdims=True), axis=0, keepdims=True)
            x_std[np.where(x_std < 1e-10)] = 1.0  # suppress values where std = 0.0
            self._std = x_std

        data = (data - self._mean) / self._std
        return data

    def unnormalize(self, data):
        """
        Reverts the effect of the normalization.
        :param data: An np-array containing the normalized data.
        :return: An np-array containing the unnormalized data.
        """
        if self._std is None or self._mean is None:
            raise ValueError('mean and std not set, cannot unnormalize')
        return data*self._std + self._mean

    def load(self, save_path):
        """
        Restores internal parameters from `.npz` file found in `save_path`.
        :param save_path: Path to where the `.npz` file for this normalizer is located.
        """
        params = np.load(os.path.join(save_path, self.get_name() + '.npz'))
        self._mean = params['Xmean']
        self._std = params['Xstd']

    def save(self, save_path, override_name=None):
        """
        Saves the name of the normalizer into a text file named `normalizer.txt`. Also saves the parameters used into
        `save_path` as a `.npz` file.
        :param save_path: Where to save text file and parameters to.
        """
        save_loc = self.get_save_location(save_path, override_name)
        np.savez_compressed(save_loc, Xmean=self._mean, Xstd=self._std)


class MeanTanhNormalizer(Normalizer):
    """
    Composite Normalizer that first normalizes to N(0, 1) and then squashes it to the interval [-1, 1].
    """
    def __init__(self, mean=None, std=None, x_min=None, x_max=None):
        self._mean_normalizer = MeanNormalizer(mean=mean, std=std)
        self._tanh_normalizer = TanhNormalizer(x_min=x_min, x_max=x_max)

    def normalize(self, data, force_recomputation=False):
        """
        First maps data joint-wise to N(0, 1) then to [-1, 1]. 

        :param data: an np-array in the format (nr_data_points, input_dim, window_size)
        :param force_recomputation: ignored
        :return: the standardized data
        """
        data_mean = self._mean_normalizer.normalize(data)
        data_tanh = self._tanh_normalizer.normalize(data_mean)
        return data_tanh

    def unnormalize(self, data):
        """
        Reverts the effect of the normalization.
        :param data: An np-array containing the normalized data.
        :return: An np-array containing the unnormalized data.
        """
        data_tanh = self._tanh_normalizer.unnormalize(data)
        data_mean = self._mean_normalizer.unnormalize(data_tanh)
        return data_mean

    def load(self, save_path):
        """
        Restores internal parameters from `.npz` file found in `save_path`.
        :param save_path: Path to where the `.npz` file for this normalizer is located.
        """
        self._tanh_normalizer.load(save_path)
        self._mean_normalizer.load(save_path)

    def save(self, save_path, override_name=None):
        """
        Saves the name of the normalizer into a text file named `normalizer.txt`. Also saves the parameters used into
        `save_path` as a `.npz` file.
        :param save_path: Where to save text file and parameters to.
        """
        self._tanh_normalizer.save(save_path, self.get_name())
        self._mean_normalizer.save(save_path, self.get_name())
