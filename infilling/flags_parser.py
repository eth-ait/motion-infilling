"""
Define some common command line options and utility functions to parse those options
"""
import ast
import collections
import os

from tbase.normalizers import MeanNormalizer
from tbase.normalizers import TanhNormalizer
from tbase.normalizers import MeanTanhNormalizer
from tbase.normalizers import SigmoidNormalizer
from tbase.perturbator import BlockPerturbator
from tbase.perturbator import ColumnPerturbator
from tbase.perturbator import ColumnPerturbatorGaussian
from tbase.perturbator import ColumnAndJointPerturbator
from tbase.perturbator import Curriculum
from tbase.perturbator import CombinedCurriculum


def define_checkpoint_dir(flags, default='checkpoints'):
    flags.DEFINE_string('checkpoint_dir', default, 'root directory where checkpoints are stored [{}]'.format(default))


def define_model_name(flags, default='default_model'):
    flags.DEFINE_string('name', default, 'name of the model [{}]'.format(default))


def define_run(flags, default=0):
    flags.DEFINE_integer('run', default, 'which run to restore [{}]'.format(default))


def define_seed(flags, default=42):
    flags.DEFINE_integer('seed', default, 'seed for random number generator [{}]'.format(default))


def define_if_test(flags):
    flags.DEFINE_boolean('test', False, 'if set, this is a test run')


def define_normalizer(flags, default='tanh'):
    flags.DEFINE_string('normalizer', default, 'which normalizer to use, "std", "tanh" (interval [-1, 1]) or "sigmoid" (interval [0, 1]) [{}]'.format(default))


def define_perturbator(flags, default='none'):
    flags.DEFINE_string('perturbator', default, 'which perturbator to use: "column", "block" or "none" [{}]'.format(default))


def define_perturbation_amount(flags, default=1):
    flags.DEFINE_integer('perturbation_amount', default, 'how many random masks used to corrupt input [{}]'.format(default))


def define_perturbation_size(flags, default="[60]"):
    flags.DEFINE_string('perturbation_size', default, 'sizes to be used by mask [{}]'.format(default))


def define_if_masked_loss(flags):
    flags.DEFINE_boolean('use_masked_loss', False, 'if set, calculates the loss only the corrupted portion of the input')


def define_batch_size(flags, default=10):
    flags.DEFINE_integer('batch_size', default, 'size of the mini-batches [{}]'.format(default))


def define_model(flags, default='vanilla'):
    flags.DEFINE_string('model', default, 'which model to use [{}]'.format(default))


def define_discard_foot_contacts(flags):
    flags.DEFINE_bool('discard_foot_contacts', False, 'if set, foot contact states are removed from the input data')


def get_data_path():
    """Returns the default path where the data is stored."""
    return '../data_preprocessed/'


def get_checkpoints_path():
    """Returns the default path where checkpoints are stored."""
    return '../pretrained-models'


def string_to_list(v):
    result = ast.literal_eval(v)
    if not isinstance(result, collections.Iterable) or isinstance(result, str):
        result = [result]
    return result


def create_model_path(checkpoint_dir, name, run):
    return os.path.join(checkpoint_dir, name, 'run_{:0>3}'.format(run))


def get_model_path(flags):
    return create_model_path(get_checkpoints_path(), flags.name, flags.run)


def get_normalizer(flags):
    n = flags.normalizer.lower()
    if n == 'mean':
        normalizer = MeanNormalizer
    elif n == 'tanh':
        normalizer = TanhNormalizer
    elif n == 'sigmoid':
        normalizer = SigmoidNormalizer
    elif n == 'meantanh':
        normalizer = MeanTanhNormalizer
    else:
        raise ValueError('normalizer "{}" unknown'.format(flags.normalizer))
    return normalizer


def get_perturbator(flags, is_inference=False):
    sizes = string_to_list(flags.perturbation_size)
    if flags.perturbator.lower() == 'block':
        perturbator = BlockPerturbator(sizes=sizes, amount=flags.perturbation_amount, value=0.0)
    elif flags.perturbator.lower() == 'column':
        bias_gaussian = (120, 30) if is_inference else None
        widths = [sizes] if isinstance(sizes, int) else sizes
        perturbator = ColumnPerturbator(widths=widths,
                                        amount=flags.perturbation_amount,
                                        value=0.0,
                                        bias_gaussian=bias_gaussian,
                                        is_inference=is_inference)
    else:
        perturbator = None
    return perturbator


def get_curriculum(flags):
    """
    Returns the curriculum to be used and the respective perturbator intended to be used with it.
    """
    c = flags.curriculum.lower()
    if c == 'none':
        return None, None
    else:
        params = string_to_list(c)
        if flags.add_joint_perturbator:
            curr = CombinedCurriculum(*params, max_epoch=flags.n_epochs)
            perturbator = ColumnAndJointPerturbator(curr.curriculum.start_mean_width, curr.curriculum.width_std, value=0.0)
        else:
            curr =  Curriculum(*params)
            perturbator = ColumnPerturbatorGaussian(curr.start_mean_width, curr.width_std, value=0.0)
        return curr, perturbator
