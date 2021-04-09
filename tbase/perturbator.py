import numpy as np
from tbase.skeleton import Skeleton


class Perturbator(object):
    """
    Perturbates data.
    """
    def __init__(self, value):
        self._last_mask = None
        self._value = value

    def perturbate(self, data, rng=np.random):
        """
        Perturbate the incoming data. Returns both the perturbed image as well as the mask used for the corruption
        process (entries are 1.0 if the corresponding entry was masked, 0.0 otherwise).
        :param data: An np array of batched images, i.e. of shape (batch_size, height, width).
        :param rng: A random number generator.
        :return: The perturbed image and the corresponding mask.
        """
        raise NotImplementedError()

    def update(self, **kwargs):
        """
        Can be used to update parameters of the perturbator. E.g. useful to increase the amount of corruption during
        training.
        """
        raise NotImplementedError()

    def reapply_last_perturbation(self, data):
        """
        Applies the last perturbation again. Useful to reproduce the same perturbation.
        :param data: An np array of batched images, i.e. of shape (batch_size, height, width).
        :return: The perturbed image and the corresponding mask.
        """
        if self._last_mask is None:
            raise RuntimeError('last mask is not set so cannot reapply perturbation')
        perturbed = np.copy(data)
        # make sure we're not going out of bounds when replacing
        perturbed[self._last_mask[:, :perturbed.shape[1], ...]] = self._value
        return perturbed, np.array(self._last_mask, dtype=np.float32)


class BlockPerturbator(Perturbator):
    """
    Randomly perturbates an image with rectangular block artefacts. Note that blocks can overlap.
    """
    def __init__(self, sizes, amount, value):
        """
        Constructor.
        :param sizes: List of (height, width)-tuples representing block-sizes used for the perturbation. Sizes are
           selected uniformly at random.
        :param amount: How many block artefacts to introduce.
        :param value: The value that will be used to fill the perturbed region.
        """
        super().__init__(value)
        self._sizes = np.array(sizes)
        self._amount = amount

    def perturbate(self, data, rng=np.random):
        perturbed = np.copy(data)
        mask = np.zeros(data.shape, dtype=np.bool)
        batch_size = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]

        for a in range(self._amount):
            for b in range(batch_size):
                # choose a size of the block artefact at random
                size = self._sizes[rng.randint(0, len(self._sizes))]

                # choose a random location in the image
                start_y = rng.choice(height)
                start_x = rng.choice(width)

                end_y = min(start_y + size[0], height)
                end_x = min(start_x + size[1], width)

                mask[b, start_y:end_y, start_x:end_x] = True

        perturbed[mask] = self._value
        self._last_mask = mask
        return perturbed, np.array(mask, dtype=np.float32)


class ColumnPerturbator(Perturbator):
    """
    Randomly masks whole columns of a given width in the input image. Tries to make all masks non-overlapping. This
    can be difficult if many masks are requested and the random position is biased to a narrow gaussian.
    """
    def __init__(self, widths, amount, value, bias_gaussian=None, is_inference=False):
        """
        Constructor.
        :param widths: List of column widths.
        :param amount: How many artefacts to introduce.
        :param value:  The value that will be used to fill the perturbed region.
        :param bias_gaussian: If set, the position of the mask will be sampled according to the given
          1D Gaussian. Must be a two tuple of the form (mean, standard deviation)
        """
        super().__init__(value)
        assert bias_gaussian is None or len(bias_gaussian) == 2
        self._widths = np.array(widths)
        self._amount = amount
        self._bias_mean = bias_gaussian[0] if bias_gaussian is not None else None
        self._bias_std = bias_gaussian[1] if bias_gaussian is not None else None
        self._is_inference = is_inference

    def get_mask(self, data_shape, rng):
        mask = np.zeros(data_shape, dtype=np.bool)
        batch_size = data_shape[0]
        img_width = data_shape[2]

        for b in range(batch_size):
            masked_cols = set()

            for a in range(self._amount):
                # choose a width at random
                size = self.choose_random_width(rng)

                # choose a random location in the image
                if self._is_inference:
                    mid = img_width//2
                    start_x = mid - size//2
                    end_x = start_x + size
                else:
                    start_x, end_x = self.choose_random_position(rng, size, img_width)

                # make sure no overlaps occur
                max_tries = 100
                while (start_x in masked_cols or end_x-1 in masked_cols) and max_tries > 0:
                    start_x, end_x = self.choose_random_position(rng, size, img_width)
                    max_tries -= 1

                if max_tries <= 0:
                    # could not find a non-overlapping block, so just continue
                    continue
                else:
                    masked_cols |= set(range(start_x, end_x))
                    mask[b, :, start_x:end_x] = True

        return mask

    def perturbate(self, data, rng=np.random):
        perturbed = np.copy(data)
        mask = self.get_mask(data_shape=data.shape, rng=rng)
        perturbed[mask] = self._value
        self._last_mask = mask
        return perturbed, np.array(mask, dtype=np.float32)

    def update(self, **kwargs):
        new_widths = kwargs['new_widths']
        new_widths = new_widths if isinstance(new_widths, list) else [new_widths]
        self._widths = np.array(new_widths)

    def choose_random_width(self, rng):
        return rng.choice(self._widths)

    def choose_random_position(self, rng, size, img_width):
        if self._bias_mean is None or self._bias_std is None:
            # choose a random location in the image uniform at random
            start_x = rng.choice(img_width - size)
            end_x = min(start_x + size, img_width)  # technically not necessary, but do it anyway to be sure
        else:
            # sample from a Gaussian
            loc = rng.normal(self._bias_mean, self._bias_std)
            loc = int(np.round(loc)) - 1
            start_x = max(0, loc - size // 2)
            end_x = min(start_x + size, img_width)
        return start_x, end_x


class ColumnPerturbatorGaussian(ColumnPerturbator):
    """
    A column perturbator that samples the width of the perturbation from a Gaussian.
    """
    def __init__(self, width_mean, width_std, value):
        super().__init__([width_mean], 1, value)
        self._width_mean = width_mean
        self._width_std = width_std

    def choose_random_width(self, rng):
        return int(np.round(rng.normal(self._width_mean, self._width_std)))

    def update(self, **kwargs):
        self._width_mean = kwargs['new_mean']
        self._width_std = kwargs['new_std']


class ColumnAndJointPerturbator(ColumnPerturbatorGaussian):
    """
    Randomly masks whole columns or rows (corresponding to joints)
    """
    def __init__(self, width_mean, width_std, value):
        super().__init__(width_mean, width_std, value)
        self._nr_joints = 1  # nr of joints that are suppressed
        self._all_joints = np.arange(1, len(Skeleton.ALL_JOINTS))  # without root

    def get_mask_joints(self, data_shape, rng):
        mask = np.zeros(data_shape, dtype=np.bool)
        batch_size = data_shape[0]

        for b in range(batch_size):
            # choose required amount of random joints
            rng.shuffle(self._all_joints)
            joint_idxs = self._all_joints[:self._nr_joints]

            for j in joint_idxs:
                mask[b, j*3:j*3+3, :] = True

        return mask

    def perturbate(self, data, rng=np.random):
        perturbed = np.copy(data)

        if bool(rng.binomial(1, 0.7)):
            # do column perturbation
            mask = self.get_mask(data_shape=data.shape, rng=rng)
        else:
            # do row perturbation
            mask = self.get_mask_joints(data_shape=data.shape, rng=rng)

        perturbed[mask] = self._value
        self._last_mask = mask
        return perturbed, np.array(mask, dtype=np.float32)

    def update(self, **kwargs):
        self._width_mean = kwargs['new_mean']
        self._width_std = kwargs['new_std']
        self._nr_joints = kwargs['new_nr_joints']


class Curriculum(object):
    """
    Represents a learning curriculum for a column perturbator that samples the width of the perturbation from
    a Gaussian.
    """
    def __init__(self, start_mean_width, end_mean_width, width_std, increase_every, increase_amount, start_epoch):
        """
        Constructor.
        :param start_mean_width: Mean of the number of consecutively missing columns in the beginning. 
        :param end_mean_width: Mean of the number of concescutively missing columns in the end.
        :param width_std: Standard deviation for the width.
        :param increase_every: When to increase the current 'width' (e.g. every second epoch).
        :param increase_amount: By how much to increase the current 'width' if the update is due.
        :param start_epoch: At which epoch to start increasing (0-based)
        """
        self.start_mean_width = start_mean_width
        self.end_mean_width = end_mean_width
        self.width_std = width_std
        self.increase_every = increase_every
        self.increase_amount = increase_amount
        self.start_epoch = start_epoch

    def get_new_params(self, current_epoch):
        """
        Returns the Gaussian to be used for the given (0-based) epoch. 
        """
        r_vals = {'new_mean': self.start_mean_width, 'new_std': self.width_std}
        epochs_passed = current_epoch - self.start_epoch
        if epochs_passed >= 0:
            n_incs = (epochs_passed + 1) // self.increase_every
            current_width = n_incs * self.increase_amount
            current_width = min(self.end_mean_width, self.start_mean_width + current_width)
            r_vals['new_mean'] = current_width
            r_vals['new_std'] = self.width_std
        return r_vals


class CombinedCurriculum(object):
    def __init__(self, start_mean_width, end_mean_width, width_std, increase_every, increase_amount, start_epoch,
                 max_epoch):
        self.curriculum = Curriculum(start_mean_width,
                                     end_mean_width,
                                     width_std,
                                     increase_every,
                                     increase_amount,
                                     start_epoch)
        self.max_nr_joints = 3
        self.increase_joint_after = max_epoch // self.max_nr_joints

    def get_new_params(self, current_epoch):
        c_vals = self.curriculum.get_new_params(current_epoch)
        nr_joints = min(int(np.ceil(current_epoch/self.increase_joint_after)), self.max_nr_joints)
        r_vals = c_vals.copy()
        r_vals.update({'new_nr_joints': nr_joints})
        return r_vals


if __name__ == '__main__':
    p = ColumnAndJointPerturbator(5, 2, 0)
    toy_data = np.reshape(np.arange(10*5*18), [10, 5, 18])
    rng = np.random.RandomState(500)
    pert, mask = p.perturbate(toy_data, rng=rng)
    for x in pert:
        print(x)

    print('*****************')

    p.update(new_mean=3, new_std=1)
    rng = np.random
    pert, mask = p.perturbate(toy_data, rng=rng)
    for x in pert:
        print(x)