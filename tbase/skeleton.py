import numpy as np
import quaternion

from tbase.shader import Shader
from tbase import utils
from tbase.utils import Quaternion
try:
    from pyglet.gl import *
except:
    print("WARNING: pyglet cannot be imported but might be required for visualization.")

VERTEX_SHADER = ['''
varying vec3 normal, lightDir0, lightDir1, eyeVec;

void main()
{
    normal = gl_NormalMatrix * gl_Normal;

    vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

    lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex);
    lightDir1 = vec3(gl_LightSource[1].position.xyz - vVertex);
    eyeVec = -vVertex;

    gl_Position = ftransform();
}
''']
FRAGMENT_SHADER = ['''
varying vec3 normal, lightDir0, lightDir1, eyeVec;

void main (void)
{
    vec4 final_color =
    (gl_FrontLightModelProduct.sceneColor * gl_FrontMaterial.ambient) +
    (gl_LightSource[0].ambient * gl_FrontMaterial.ambient) +
    (gl_LightSource[1].ambient * gl_FrontMaterial.ambient);

    vec3 N = normalize(normal);
    vec3 L0 = normalize(lightDir0);
    vec3 L1 = normalize(lightDir1);

    float lambertTerm0 = dot(N,L0);
    float lambertTerm1 = dot(N,L1);

    if(lambertTerm0 > 0.0)
    {
        final_color += gl_LightSource[0].diffuse *
                       gl_FrontMaterial.diffuse *
                       lambertTerm0;

        vec3 E = normalize(eyeVec);
        vec3 R = reflect(-L0, N);
        float specular = pow( max(dot(R, E), 0.0),
                         gl_FrontMaterial.shininess );
        final_color += gl_LightSource[0].specular *
                       gl_FrontMaterial.specular *
                       specular;
    }
    if(lambertTerm1 > 0.0)
    {
        final_color += gl_LightSource[1].diffuse *
                       gl_FrontMaterial.diffuse *
                       lambertTerm1;

        vec3 E = normalize(eyeVec);
        vec3 R = reflect(-L1, N);
        float specular = pow( max(dot(R, E), 0.0),
                         gl_FrontMaterial.shininess );
        final_color += gl_LightSource[1].specular *
                       gl_FrontMaterial.specular *
                       specular;
    }
    gl_FragColor = final_color;
}
''']


def _correct_rotational_difference(points, target_dir):
    """
    Rotates all points in `points` around the z-axis according to an angle computed such that the first forward
    vector of the root is rotated onto the `target_dir` vector.
    :param points: an np array of size (nr_points, 3, nr_frames), the first point in the first frame is considered root
    :param target_dir: a 2-D vector specifying the target dir on the x/y plane
    :return: the corrected points in the same format as the input
    """
    # compute the rotation to rotate points by angle specified by target_dir
    actual_dir = points[0, :, 1] - points[0, :, 0]
    rot = utils.rotation_between(target_dir, actual_dir)

    # project all points to x-y-plane
    projected = np.copy(points)
    ori_z = np.copy(projected[:, 2:3, :])
    projected[:, 2:3, :] = np.zeros(shape=[points.shape[0], 1, points.shape[2]])

    # apply the rotation to every vector
    projected = np.reshape(np.transpose(projected, [0, 2, 1]), [-1, 3])
    rot_mult = np.expand_dims(rot, axis=0)
    proj_mult = np.expand_dims(projected, axis=-1)
    proj_corrected = np.matmul(rot_mult, proj_mult)

    # now transform back to original shape
    proj_corrected = np.reshape(np.squeeze(proj_corrected, axis=-1), [points.shape[0], points.shape[2], 3])
    proj_corrected = np.transpose(proj_corrected, [0, 2, 1])

    # restore the old z-values
    proj_corrected[:, 2:3, :] = ori_z
    return proj_corrected


def to_global_batched(points, override_trajectory=None, override_root=None):
    """
    :param points: A np array of shape (batch_size, dof, seq_lenth) where dof is assumed to contain the trajectory
      in the last three dimensions.
    :param override_trajectory: A np array of shape (batch_size, 3, seq_lenth) if the trajectory of `point` is
      to be overriden with this value.
    """
    n = points.shape[0]
    seq_len = points.shape[-1]
    body_dim = len(Skeleton.ALL_JOINTS)*3
    body_joints = points[:, :body_dim].reshape([n, -1, 3, seq_len])  # (N, n_joints, 3, seq_len)
    trajectory = points[:, body_dim:body_dim + 3]  # sometimes there's foot contacts (N, 3, seq_len)

    if override_trajectory is not None:
        trajectory = override_trajectory

    if override_root is not None:
        body_joints[:, 0] = override_root

    body_global = []
    for i in range(n):
        bg = utils.to_global(body_joints[i], trajectory[i].T)
        body_global.append(bg)

    return np.array(body_global) * Skeleton.TO_CM  # (N, n_joints, 3, seq_len)


class Skeleton(object):
    """
    Defines a skeleton.
    """
    # Making joint indices explicit
    ALL_JOINTS = list(range(0, 22))
    ROOT, HIP, RIGHT_GROIN, RIGHT_KNEE, RIGHT_HEEL, RIGHT_TOE, \
    LEFT_GROIN, LEFT_KNEE, LEFT_HEEL, LEFT_TOE, \
    LOWER_BODY, UPPER_BODY, NECK, HEAD, \
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HAND, \
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HAND = ALL_JOINTS

    # Indices pointing to the parent for every joint in the skeleton, -1 if it has no parent
    PARENTS = np.array([-1, ROOT, HIP, RIGHT_GROIN, RIGHT_KNEE, RIGHT_HEEL, HIP, LEFT_GROIN, LEFT_KNEE, LEFT_HEEL, HIP,
                        LOWER_BODY, UPPER_BODY, NECK, NECK, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
                        NECK, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST])

    # Indices defining the actual bones (i.e. start to end joint) in the skeleton. Note that it starts from 2 because
    # the link between joint at index 1 and 0 is a straight line from the hip to the root located on the floor, which
    # is not a bone.
    BONES = list(zip(list(range(2, len(PARENTS))), PARENTS[2:]))

    # Number of bones in the skeleton
    N_BONES = len(BONES)

    # Convert the data to centimeters (this is a constant taken from Holden's code).
    TO_CM = 1 / 0.16

    # Length of each bone
    BONE_LENGTHS = np.array([2.40, 7.15, 7.49, 2.36, 2.37, 7.43, 7.50, 2.41, 2.04, 2.05, 1.75, 1.76, 2.90, 4.98,
                             3.48, 0.71, 2.73, 5.24, 3.44, 0.62])

    # Indices pointing to the feet. Note that those indices are only valid if the data is NOT flattened, i.e. it is
    # e.g. in the form (nr_joints, 3). Also note that the order in the `FEET` vector is chosen such that it is the same
    # as the foot contact states encoded in the training data.
    FEET = np.array([RIGHT_HEEL, RIGHT_TOE, LEFT_HEEL, LEFT_TOE])

    # For convenience: Indices of the feet for when the data is flattened. E.g. the data used as input to the motion
    # manifold trainig procedure is flattened (vector of length 73 where the first 66 elements are the joint positions).
    # Rows correspond to the indices for retrieving the position as a 3D vector.
    FEET_FLATTENED = np.array([list(range(i, i+3)) for i in FEET*3])

    @classmethod
    def idx_to_foot(cls, idx):
        return cls.FEET[idx]

    @classmethod
    def foot_to_idx(cls, foot):
        return np.where(cls.FEET == foot)[0][0]


class SkeletonSequence(object):
    """
    An animated skeleton, i.e. a sequence of skeletons over time.
    """
    _counter = 0

    def __init__(self, sequence, x_offset=None, offsets=None, target_dir=None, name=None, color=None, interp=None,
                 static_frames=None):
        """
        Constructor
        :param sequence: an np array of size (dim, nr_frames) where the first axis contains the information about the
           skeleton in the following order: [...3D joints..., x-velocity, y-velocity, rotational velocity,
           foot contacts]. Usually there are 22 joints, and 4 entries for the foot contacts state making this a vector
           of size 73.
        :param x_offset: an offset on the x-axis to be applied when drawing the sequence. This is useful when several
          sequences are shown in parallel.
        :param offsets: a 3 dimensional np array specifiying (x, y, z)-offsets to be applied when drawing the sequence
        :param target_dir: a 2 dimensional vector on the floor which is used to compute a correction angle around the
          z-axis.
        :param name: string, a name for this sequence
        :param color: 3-tuple, a color for this sequence used for visualizations
        :param interp: list of indices specifying for which frames the motion was filled in (interpolated or inpainted),
          only used for visualization
        :param static_frames: list of frame indices that should be drawn statically
        """
        assert isinstance(sequence, np.ndarray), 'Input sequence must be an np array.'
        assert len(sequence.shape) == 2, 'Input expected in format (dim, nr_frames)'

        # make a copy because we might be changing stuff internally
        data = sequence.copy()

        # foot contacts might not be available
        self.feet_available = sequence.shape[0] > 69
        pose_dim = len(Skeleton.ALL_JOINTS)*3

        self._nr_frames = len(data[0])
        self._joints = data[:pose_dim, :]
        self._joints = self._joints.reshape(-1, 3, self._nr_frames)  # now in format (nr_joints, 3, nr_frames)
        self._root_x = data[pose_dim + 0, :]
        self._root_z = data[pose_dim + 1, :]
        self._root_r = data[pose_dim + 2, :]
        self._feet = data[-4:, :] if self.feet_available else None
        self._frame_pointer = 0
        self._linkage = None
        self._drawing_offset = [0.0, 0.0, 0.0] if offsets is None else offsets
        self._x_offset = x_offset or 0.0
        self._interp = set(interp) if isinstance(interp, np.ndarray) or interp else None

        self._joints = utils.to_global(self._joints, np.stack([self._root_x, self._root_z, self._root_r], axis=1))

        # swap y-z axis
        temp = np.copy(self._joints[:, 1, :])
        self._joints[:, 1, :] = self._joints[:, 2, :]
        self._joints[:, 2, :] = temp

        # correct for rotational discrepancy around z-axis
        if target_dir is not None:
            self._joints = _correct_rotational_difference(self._joints, target_dir)

        # define boolean if this sequence should be higlighted
        self.highlight = False

        # list of frame indices that should be drawn statically
        self._static_frames = static_frames

        # list of points that are drawn statically (for end effector visualization)
        self._static_points = None

        # boolean that indicates if linkage must be recomputed e.g. when underlying joints changed
        self._update_linkage = False

        self._name = name or 'SkeletonSequence %d' % SkeletonSequence._counter
        self._color = color or utils.COLORS[SkeletonSequence._counter % len(utils.COLORS)] + (1,)
        self._shader = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
        SkeletonSequence._counter += 1

    def _construct_linkage(self):
        linkage = np.zeros([0, 6, self._nr_frames])
        parents = Skeleton.PARENTS
        for j in range(len(parents)):
            if parents[j] == -1:
                continue
            x = np.concatenate([self._joints[j, :, :], self._joints[parents[j], :, :]])
            linkage = np.concatenate([linkage, np.expand_dims(x, 0)])
        return linkage

    def _get_frame_at(self, frame_idx):
        joints = self.joints[:, :, frame_idx]
        links = self.linkage[:, :, frame_idx]
        return joints, links

    def _get_current_frame(self):
        return self._get_frame_at(self.frame_pointer)

    def _to_drawable_vertex(self, v):
        return [v[0] + self._x_offset + self._drawing_offset[0],
                v[1] + self._drawing_offset[1],
                v[2] + self._drawing_offset[2]]

    def _draw_cylinder(self, orientation, t, height=1.0, radius=.5, slices=32, loops=32):
        """Draws a cylinder whose primary axis is directed according to the 3D vector given by `orientation` and which
        is translated according to `t`."""
        # gluCylinder draws a cylinder at the origin pointing upward the z axis. Thus find the necessary rotation
        # to rotate the default cylinder into the orientation specified.
        v1 = np.array([[0.0, 0.0, height]])
        q = Quaternion.rotate_from_to(v1, orientation[np.newaxis, ...])
        rot = quaternion.as_rotation_matrix(q)

        # save current modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # first apply translation
        _t = [t[0], t[1], t[2]]
        glTranslated(*_t)

        # then apply rotation
        gl_list = utils.build_gl_rot_matrix(rot[0].T)
        glMultMatrixd(gl_list)

        # draw cylinder and base disk
        quadratic = glu.gluNewQuadric()
        glu.gluCylinder(quadratic, radius, radius, 1.0 * height, slices, loops)
        glu.gluDisk(quadratic, 0.0, radius, slices, loops)

        # draw top disk
        _t = [v1[0, 1], v1[0, 1], v1[0, 2]]
        glTranslated(*_t)
        glu.gluDisk(quadratic, 0.0, radius, slices, loops)

        # restore previous modelviewmatrix
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    @property
    def joints(self):
        return self._joints

    @property
    def nr_frames(self):
        return self._nr_frames

    @property
    def linkage(self):
        if self._linkage is None or self._update_linkage:
            self._linkage = self._construct_linkage()
            self._update_linkage = False
        return self._linkage

    @property
    def frame_pointer(self):
        return self._frame_pointer

    @frame_pointer.setter
    def frame_pointer(self, value):
        self._frame_pointer = value

    def frame_pointer_clipped(self):
        ptr = max(0, self.frame_pointer)
        ptr = min(ptr, self.nr_frames - 1)
        return ptr

    @property
    def name(self):
        return self._name

    @property
    def color(self):
        return self._color

    def set_static_frames(self, idxs):
        self._static_frames = idxs

    def set_static_points(self, points):
        # points has shape [n_points, 3]
        self._static_points = points

    def get_root_trajectory(self):
        return self.joints[0, :, :]

    def export_to_csv(self, filename=None):
        """
        Export 3D joint positions and skeleton definition to a CSV file.
        :param filename: CSV target file or the name of this sequence if None
        """
        if filename is None:
            filename = self._name + '.txt'

        with open(filename, 'wb') as f:
            # first row is the definition of the skeleton
            np.savetxt(f, np.expand_dims(Skeleton.PARENTS, 0), delimiter=',', fmt='%d')

            # second row are the bones (stored for convenience)
            np.savetxt(f, np.reshape(np.array(Skeleton.BONES), [1, -1]), delimiter=',', fmt='%d')

            # third row is frame indices that are infilled
            infilled = np.expand_dims(np.array(list(self._interp)), 0) if self._interp is not None else np.array([-1], dtype=np.int64)
            np.savetxt(f, infilled, delimiter=',', fmt='%d')

            # next `n_frames` rows is the motion
            n_frames = self._joints.shape[-1]
            joints = np.reshape(self._joints, [-1, n_frames])
            np.savetxt(f, joints, delimiter=',')

    def rotate_around_z_axis(self, angle):
        """
        Rotate the whole sequence around the z axis of the first pose in the sequence for the specified angle.
        """
        rotation = Quaternion.from_angle_axis(np.array([[0, 0, 1]]), angle)

        # temporarily put whole sequence such that root of first pose is in the origin
        translation = np.repeat(np.copy(self._joints[0:1, :, 0:1]), self._joints.shape[0], axis=0)
        translation[:, 2, :] = 0.0
        joints_ori = self._joints - translation
        self._joints = joints_ori

        # now rotate the sequence in the origin
        for f in range(joints_ori.shape[2]):
            joints_ori[:, :, f] = Quaternion.apply_rotation_to(rotation, joints_ori[:, :, f])

        # restore original translation
        self._joints = joints_ori + translation
        self._update_linkage = True

    def translate_along_axis(self, axis, t):
        axis_idx = 'xy'.index(axis)
        for f in range(self._joints.shape[2]):
            self._joints[:, axis_idx, f] = self._joints[:, axis_idx, f] + t
        self._update_linkage = True

    def set_color(self):
        if self.highlight or self.is_infilling():
            color = utils.lighten_color(self.color, 0.5)
        else:
            color = self.color

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, utils.vec(*color))
        glColor4f(*color)

    def is_infilling(self):
        return self._interp and self.frame_pointer in self._interp

    def draw_timestep(self, timestep, draw_cylinders=True):
        self.set_color()
        joints, links = self._get_frame_at(timestep)

        self._shader.bind()

        if draw_cylinders:
            for bone in links[1:, :]:
                end_vertex = self._to_drawable_vertex(bone[3:6])
                bone_dir = bone[0:3] - bone[3:6]
                bone_len = np.sqrt(np.sum((np.multiply(bone_dir, bone_dir))))
                if bone_len < 1e-8:
                    continue
                self._draw_cylinder(orientation=bone_dir, t=end_vertex, height=bone_len)
        else:
            for joint in joints[1:, :]:
                glBegin(GL_POINTS)
                vertex = self._to_drawable_vertex(joint)
                glVertex3f(*vertex)
                glEnd()

            for bone in links[1:, :]:
                glBegin(GL_LINES)
                vertex1 = self._to_drawable_vertex(bone[0:3])
                vertex2 = self._to_drawable_vertex(bone[3:6])
                glVertex3f(*vertex1)
                glVertex3f(*vertex2)
                glEnd()

        self._shader.unbind()

    def draw_current_timestep(self, draw_cylinders=True):
        self.draw_timestep(self.frame_pointer, draw_cylinders=draw_cylinders)

    def draw_static_frames(self, draw_cylinders=True):
        if self._static_frames is not None:
            [self.draw_timestep(idx, draw_cylinders) for idx in self._static_frames]

    def draw_static_points(self):
        if self._static_points is None:
            return

        static_color = utils.WHITE
        glColor3f(*static_color)
        for v in self._static_points:
            v_d = self._to_drawable_vertex(v)
            glBegin(GL_POINTS)
            glVertex3f(*v_d)
            glEnd()

    def draw_root_trajectory(self, full=False):
        glColor4f(*self.color)
        if full:
            root_trajectory = self.joints[0, :, :]
        else:
            start = max(0, self.frame_pointer_clipped() - 200)
            end = min(self._nr_frames, self.frame_pointer_clipped() + 200)
            root_trajectory = self.joints[0, :, start:end]
        for i in range(len(root_trajectory[0])):
            vertex = self._to_drawable_vertex(root_trajectory[:, i])
            glBegin(GL_POINTS)
            glVertex3f(*vertex)
            glEnd()

    def draw_foot_contacts(self, part=None):
        """
        Draws a line for each part of the feet (heel and toe) when it is on the ground according to the foot
        contact information supplied with the original data.
        :param part: an index from Skeleton.FEET indicating which part should be displayed or None if all parts are
          to be displayed
        """
        if not self.feet_available:
            return

        ori_color = self.color
        bright_color = utils.lighten_color(self.color, 0.5)

        if part is None:
            show_parts = Skeleton.FEET
        else:
            show_parts = [part]

        contact_info = self._feet
        joints = self.joints

        def is_on_ground(frame_id, foot_part):
            return contact_info[Skeleton.foot_to_idx(foot_part), frame_id] > 0.5

        for idx in show_parts:
            positions = joints[idx, :, :]
            for i in range(0, positions.shape[-1]-1):
                v1 = self._to_drawable_vertex(positions[:, i])
                v2 = self._to_drawable_vertex(positions[:, i+1])

                if is_on_ground(i, idx):
                    if is_on_ground(i+1, idx):
                        glColor4f(*bright_color)
                        glBegin(GL_LINES)
                        glVertex3f(*v1)
                        glVertex3f(*v2)
                        glEnd()
                    else:
                        glColor4f(*bright_color)
                        glBegin(GL_POINTS)
                        glVertex3f(*v1)
                        glEnd()

                        glColor4f(*ori_color)
                        glBegin(GL_POINTS)
                        glVertex3f(*v2)
                        glEnd()
                else:
                    if is_on_ground(i+1, idx):
                        glColor4f(*ori_color)
                        glBegin(GL_POINTS)
                        glVertex3f(*v1)
                        glEnd()

                        glColor4f(*bright_color)
                        glBegin(GL_POINTS)
                        glVertex3f(*v2)
                        glEnd()
                    else:
                        glColor4f(*ori_color)
                        glBegin(GL_LINES)
                        glVertex3f(*v1)
                        glVertex3f(*v2)
                        glEnd()

    def fast_forward(self):
        self.frame_pointer += 1
        if self.frame_pointer >= self.nr_frames:
            self.frame_pointer = 0

    def rewind(self):
        self.frame_pointer -= 1
        if self.frame_pointer < 0:
            self.frame_pointer = self.nr_frames-1

    def reset_time(self):
        self.frame_pointer = -1
