import datetime
import os
import subprocess
import numpy as np
import tensorflow as tf
import time
import quaternion

try:
    from pyglet.gl import *
except:
    print("WARNING: pyglet cannot be imported but might be required for visualization.")

from scipy.ndimage import filters as filters


BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0.2, 0.2)
YELLOW = (1, 1, 0.2)
ORANGE = (1, 0.7, 0.2)
GREEN = (0.2, 0.9, 0.2)
BLUE = (0.2, 0.3, 0.9)
PURPLE = (1.0, 0, 1.0)
CRIMSON = (220.0/255.0, 20.0/255.0, 60.0/255.0)
COLORS = (BLUE, GREEN, ORANGE, RED, YELLOW, PURPLE, CRIMSON)


def get_latest_checkpoint(model_path):
    """
    Get the latest checkpoint file from the model_path.
    :param model_path: String pointing to something like /path/to/trained/models/model_name/run_003
    :return: The path to the latest checkpoint saved for this model.
    """
    ckpt = tf.train.get_checkpoint_state(model_path, latest_filename=None)
    if ckpt and ckpt.model_checkpoint_path:
        # prepend the path of `model_path` thus replacing the one stored in the model as the files might have been moved
        ckpt_path = ckpt.model_checkpoint_path
        # because we have models trained on linux and windows, but both should be able to run on other OSes as well,
        # can't just split by os.path.sep in the following
        sp = ckpt_path.split(os.path.sep)
        if '/' in sp[-1]:
            # this was a model trained on windows but now we are on linux
            file_name = sp[-1].split('/')[-1]
        elif '\\' in sp[-1]:
            # this was a model trained on linux but now we are on windows
            file_name = sp[-1].split('\\')[-1]
        else:
            # we're on the same platform as we trained the model on
            file_name = sp[-1]
        return os.path.join(model_path, file_name)
    return None


def vec(*args):
    """Create ctype arrays of floats."""
    return (GLfloat * len(args))(*args)


def build_gl_rot_matrix(rot):
    """Builds a 4-by-4 rotation matrix from the 3-by-3 rotation matrix `rot`. The result can be used in calls to OpenGL
     functions."""
    m = (GLdouble * 16)()
    m[0] = rot[0, 0]
    m[1] = rot[0, 1]
    m[2] = rot[0, 2]
    m[3] = GLdouble(0.0)

    m[4] = rot[1, 0]
    m[5] = rot[1, 1]
    m[6] = rot[1, 2]
    m[7] = GLdouble(0.0)

    m[8] = rot[2, 0]
    m[9] = rot[2, 1]
    m[10] = rot[2, 2]
    m[11] = GLdouble(0.0)

    m[12] = GLdouble(0.0)
    m[13] = GLdouble(0.0)
    m[14] = GLdouble(0.0)
    m[15] = GLdouble(1.0)
    return m


def test_installation():
    """Simple test to check if installation was successful. Should print '[4 3 1 2]' to the console."""
    x = tf.constant(np.array([3, 2, 0, 2], dtype=np.int64))
    op = tf.add(x, tf.constant([1, 1, 1, 1], dtype=tf.int64))
    with tf.Session() as sess:
        print(sess.run(op))


def get_current_hg_revision():
    """Returns the current hg revision of the current working directory."""
    try:
        pipe = subprocess.Popen(['hg', '--debug', 'id', '-i'], stdout=subprocess.PIPE)
        return pipe.stdout.read()
    except OSError or ValueError:
        return 'Could not retrieve revision'


def to_printable_string(**kwargs):
    "Puts all keyword-value pairs into a printable string."
    s = ''
    for k, v in kwargs.items():
        s += '{}: {}\n'.format(k, v)
    return s


def dump_configuration(tags, target_dir):
    """
    Creates a file 'config.txt' in `target_dir` which contains all key-value pairs found in the given `tags` namespace
    as well as the current hg revision and the date of creation. The dumped file is human readable.
    :param tags: a namespace that is to be dumped
    :param target_dir: the directory into which to dump the configuration
    """
    if not os.path.isdir(target_dir):
        raise ValueError("'%s' is not a valid directory" % target_dir)

    file_name = os.path.join(target_dir, 'config.txt')
    with open(file_name, 'w') as f:
        for k, v in vars(tags).items():
            f.write('%s: %s%s' % (k, v, os.linesep))
        f.write(os.linesep)
        f.write('hg revision: %s' % (get_current_hg_revision()))
        now = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')
        f.write('mka, %s' % now)


def get_config_entry(path, entry_key):
    """
    Searches for the entry "entry_key: value" in the file "path/config.txt" and returns the associated value. Returns
    None if the entry was not found.
    """
    def _process_line(line):
        sp = line.split(':')
        return [s.strip() for s in sp]

    with open(os.path.join(path, 'config.txt'), 'r') as f:
        for line in f:
            content = _process_line(line)
            if len(content) > 1 and content[0].lower() == entry_key.lower():
                return content[1]
    return None


def create_dir_if_not_exists(dir_path):
    """Creates the specified directory and all its parents if it does not exist yet."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_trainable_variable_by_name(name):
    """Retrieves the trainable variable with the specified name from the default graph."""
    found = [v for v in tf.trainable_variables() if str(v.name) == name+':0']
    return found[0]


def get_variables_by_name(sess, variable_names):
    """Retrieves the value of the trainable variables specified in `variable_names` from the default graph
    as np arrays and returns them in a dictionary whose keys are the names of the variable."""
    return {n: sess.run(get_trainable_variable_by_name(n)) for n in variable_names}


def numel(t):
    """Returns the number of elements in the given tensor as a tensorflow op."""
    return np.prod([k.value for k in t.get_shape()])


def rotation_between(v1, v2):
    """Returns a rotation matrix that rotates v2 around the z-axis to match v1."""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle = angle1 - angle2
    rot = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(np.array([0.0, 0.0, angle])))
    return rot


class Quaternion(object):
    """Simple helper class to deal with quaternion math."""
    @staticmethod
    def identity():
        return np.quaternion(1, 0, 0, 0)

    @staticmethod
    def rotate_from_to(v0, v1):
        """
        Computes the rotation necessary to rotate the 3-D vectors in v0 onto the 3-D vectors in v1. The actual vectors
        are expected to reside in the last dimension of v0 and v1 respectively.
        :param v0: an np array of size (..., 3)
        :param v1: an np array of size (..., 3)
        :return: quaternions that produce v1 when multiplied with v0
        """
        v0s = np.reshape(v0, [-1, 3])
        v1s = np.reshape(v1, [-1, 3])
        # compute axis of rotation
        axes = np.cross(v0s, v1s)
        # If v0 and v1 are linearly dependent, the cross product will be (0, 0, 0) which will result in no rotation at
        # all. To fix this simply choose a vector that is perpendicular to either v0 or v1 as the rotation axis
        idx = np.where((abs(axes) < np.finfo(float).eps).all(axis=1))[0]
        for i in idx:
            v = v0s[i, :]
            r = np.random.rand(1, 3)
            cross = np.cross(v, r)
            while (abs(cross) < np.finfo(float).eps).all(axis=1):
                # randomly chosen vector was linearly dependent to v, so choose another and try again
                r = np.random.rand(1, 3)
                cross = np.cross(v, r)
            # cross is non-zero and perpendicular to v0, so choose it as the rotation axis
            axes[i, :] = cross
        # compute angle between vectors (no need to correct angle because cross product
        # takes care of correct orientation)
        dot = np.sum(v0s * v1s, axis=-1)
        angle = np.arccos(dot / (np.linalg.norm(v0s, axis=-1) * np.linalg.norm(v1s, axis=-1)))
        # normalize axes
        axes /= np.linalg.norm(axes, axis=-1)[..., np.newaxis]
        qs = quaternion.from_rotation_vector(axes*angle[..., np.newaxis])
        target_shape = v0.shape[:-1]
        return np.reshape(qs, target_shape)

    @staticmethod
    def apply_rotation_to(qs, vs):
        """
        Rotate the vectors in vs elementwise according to the quaternions stored in qs. The 3-D vectors in vs are
        expected to reside in the last dimension. The product of the remaining dimensions must be equal to the flattened
        size of qs, unless it is one in which case the vector is broadcast.
        :param qs: an np array of quaternions whose flattened size is equal to the product of the leading sizes of qs
        :param vs: an np array of size (..., 3). The product of the leading dimension must match the size of qs.
        :return: the vectors in vs rotated as specified by qs and in the same shape as the input vs
        """
        vs_r = np.reshape(vs, [-1, 3, 1])
        qs_r = np.reshape(qs, [-1])
        assert vs_r.shape[0] == 1 or qs_r.shape[0] == 1 or vs_r.shape[0] == qs_r.shape[0], \
            'too many or too few quaternions supplied'
        rot = quaternion.as_rotation_matrix(qs_r)
        vs_rot = np.matmul(rot, vs_r)
        if vs_r.shape[0] == 1:
            target_shape = [qs_r.shape[0], 3]
        else:
            target_shape = vs.shape
        return np.reshape(vs_rot, target_shape)

    @staticmethod
    def mult(q1, q2):
        """Multiply arrays of quaternions element-wise."""
        if isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray):
            assert q1.shape == q2.shape, 'shapes must match when suppliying arrays of quaternions'
        return np.multiply(q1, q2)

    @staticmethod
    def compute_rotational_velocity(qs, forward, plane_axes):
        """
        Rotates the given forward vector as specified by the quaternions and then computes the rotational velocity of
        the rotated vectors around the axis which is perpendicular to the specified plane.
        :param qs: an array of quaternions that define the rotation to be applied to the forward vector
        :param forward: a 3-D np array defining the forward direction
        :param plane_axes: a 2-D tuple or list that defines the plane, e.g. [0, 2] defines the x-z-plane
        :return: a list of rotational velocities of the same length as there are quaternions in qs
        """
        assert len(plane_axes) == 2, 'need two indices to define plane'
        forward_rot = Quaternion.apply_rotation_to(qs, forward)
        # project rotated vectors onto plane
        xs = forward_rot[..., plane_axes[0]]
        ys = forward_rot[..., plane_axes[1]]
        # compute the angle between x-axis and projected point
        # NOTE: arctan2 expects ys first, but this is how Holden does it. If we switch that, all hell breaks loose.
        angles = np.arctan2(xs, ys)
        return np.reshape(angles, qs.shape)

    @staticmethod
    def conj(qs):
        """Computes the conjugate of the quaternions, i.e. negates the imaginary parts."""
        return np.conjugate(qs)

    @staticmethod
    def norm(qs):
        """Computes the length of the quaternions, i.e. the sum of the squares of the real and imaginary parts."""
        qs_r = np.reshape(qs, [-1])
        qs_arr = quaternion.as_float_array(qs_r)
        norms = np.sum(qs_arr * qs_arr, axis=-1)
        return np.reshape(norms, qs.shape)

    @staticmethod
    def invert(qs):
        """Inverts the quaternions, i.e. returns the normalized conjugates."""
        conj = quaternion.as_float_array(Quaternion.conj(qs))
        normalized = conj / Quaternion.norm(qs)[..., np.newaxis]
        return quaternion.as_quat_array(normalized)

    @staticmethod
    def from_angle_axis(axis, angle):
        """Returns a quaternion representation the rotation around the specified axis for the given angle."""
        axis_n = axis / np.linalg.norm(axis)
        return quaternion.from_rotation_vector(axis_n*angle)


def to_global(points, velocities):
    """
    Adds global transformation to the input points according to the information given by the velocities.
    :param points: An np array of 3-dimensional points in the format (nr_points, 3, sequence_length)
    :param velocities: An np array of size (sequence_length, 3), where (i, 0:2) are the velocities in the x-z-plane at
      timestep i and (i, 2) is the rotational velocity around the y-axis at timestep i
    :return: The `points` vector in the global coordinate frame
    """
    assert points.shape[-1] == velocities.shape[0], 'input dimensions of velocities and points must match'
    rotation = Quaternion.identity()
    translation = np.array([0.0, 0.0, 0.0])
    for f in range(len(points[0][0])):
        points[:, :, f] = Quaternion.apply_rotation_to(rotation, points[:, :, f])
        points[:, 0, f] = points[:, 0, f] + translation[0]
        points[:, 2, f] = points[:, 2, f] + translation[2]
        rotation = Quaternion.mult(Quaternion.from_angle_axis(np.array([[0, 1, 0]]), -velocities[f, 2]), rotation)
        trans_rot = Quaternion.apply_rotation_to(rotation, np.array([velocities[f, 0], 0, velocities[f, 1]]))
        translation += np.squeeze(trans_rot, axis=0)
    return points


def assert_tensors_equal(sess, names, values):
    """
    Checks if all tensors specified in `names` are set to a given value. If this is not the case for at least one
    of the supplied names, an assertion error is thrown.
    :param sess: the session in which the default graph is loaded
    :param names: list of strings, names of the variables whose values are to be checked
    :param values: dict of np-arrays, keys are the names of the variables
    """
    for name in names:
        np_val = sess.run(tf.get_default_graph().get_tensor_by_name(name + ':0'))
        assert np.equal(np_val, values[name]).all(), 'tensor "{0}" is not set to the expected value'.format(name)


def restore_variable(sess, name, value):
    """
    Overwrites a variable in the default graph with the given value.
    :param sess: the session in which the graph is loaded
    :param name: string, the name of the variable to be overriden
    :param value: np-array, the override-value, must match the shape of the variable
    """
    variable = tf.get_default_graph().get_tensor_by_name(name)
    sess.run(tf.assign(variable, value))


def restore_variables(sess, names, values):
    """
    Tries to locate all variables in `names` in the default graph and overwrites the current value with the value
    supplied through `values`.
    :param sess: the session in which the graph is loaded
    :param names: list of strings, names of the variables to be overriden
    :param values: dict of np-arrays, keys are names of the variables
    """
    for name in names:
        restore_variable(sess, name + ':0', values[name])


def lighten_color(color, amount):
    """
    Ligthen the color by a certain amount. Inspired by http://stackoverflow.com/questions/141855.
    :param color: a 3- or 4-tuple in range (0, 1)
    :param amount: value between (0, 1) defining how much brighter the resulting color should be
    :return: the lightened color
    """
    color_out = (min(1.0, color[0] + amount),
                 min(1.0, color[1] + amount),
                 min(1.0, color[2] + amount))
    if len(color) == 4:
        color_out += (color[3],)
    return color_out


def count_trainable_parameters():
    """Counts the number of trainable parameters in the current default graph."""
    tot_count = 0
    for v in tf.trainable_variables():
        v_count = 1
        for d in v.get_shape():
            v_count *= d.value
        tot_count += v_count
    return tot_count


def get_dir_creation_time(dir_path):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(dir_path)))


def extract_model_name_and_run(model_string):
    """
    `model_string` is a string in the format "model_name/run_id[ft]". Returns the name as a string and the run as an id.
    """
    sp = model_string.split('/')
    assert len(sp) == 2 or len(sp) == 3
    name = sp[0] if len(sp) == 2 else '{}/{}'.format(sp[0], sp[1])
    run = sp[-1]
    discard_foot_contacts = 'f' in run
    replace_traj = 't' in run
    remove = 0 + discard_foot_contacts + replace_traj
    run_id = int(run[:-remove]) if remove > 0 else int(run)
    return name, run_id, discard_foot_contacts, replace_traj


def lerp(x, y, n_samples):
    samples = np.linspace(0.0, 1.0, n_samples + 2)  # don't want x and y to be reproduced, so +2
    interpolated = np.zeros([x.shape[0], n_samples])
    for i in range(0, n_samples):
        t = samples[i + 1]
        interpolated[:, i] = x * (1.0 - t) + y * t
    return interpolated
