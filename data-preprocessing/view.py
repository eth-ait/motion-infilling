import os
import numpy as np
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from matplotlib.animation import ArtistAnimation
from mpl_toolkits.mplot3d import Axes3D
from Quaternions import Quaternions


def animation_plot(animations, interval=33.33):
    footsteps = []

    for ai in range(len(animations)):
        anim = animations[ai][0].copy()[:, 3:]

        joints, root_x, root_z, root_r = anim[:, :-7], anim[:, -7], anim[:, -6], anim[:, -5]
        joints = joints.reshape((len(joints), -1, 3))

        rotation = Quaternions.id(1)
        offsets = []
        translation = np.array([[0, 0, 0]])

        for i in range(len(joints)):
            joints[i, :, :] = rotation * joints[i]
            joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
            joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
            rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
            offsets.append(rotation * np.array([0, 0, 1]))
            translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

        animations[ai] = joints
        footsteps.append(anim[:, -4:])

    footsteps = np.array(footsteps)
    print(footsteps.shape)

    scale = 1.25 * ((len(animations)) / 2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-scale * 30, scale * 30)
    ax.set_zlim3d(0, scale * 60)
    ax.set_ylim3d(-scale * 30, scale * 30)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
    ax.set_aspect('equal')

    acolors = list(sorted(colors.cnames.keys()))[::-1]
    lines = []

    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 11, 13, 14, 15, 11, 17, 18, 19])

    for ai, anim in enumerate(animations):
        lines.append([plt.plot([0, 0], [0, 0], [0, 0], color=acolors[ai],
                               lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in
                      range(anim.shape[1])])

    def animate(i):
        changed = []
        for ai in range(len(animations)):
            offset = 25 * (ai - ((len(animations)) / 2))
            for j in range(len(parents)):
                if parents[j] != -1:
                    lines[ai][j].set_data(
                        [animations[ai][i, j, 0] + offset, animations[ai][i, parents[j], 0] + offset],
                        [-animations[ai][i, j, 2], -animations[ai][i, parents[j], 2]])
                    lines[ai][j].set_3d_properties(
                        [animations[ai][i, j, 1], animations[ai][i, parents[j], 1]])
            changed += lines

        return changed

    plt.tight_layout()

    ani = animation.FuncAnimation(fig,
                                  animate, np.arange(len(animations[0])), interval=interval)

    plt.show()


if __name__ == '__main__':

    data_path = '../data_preprocessed/valid/'
    db = 'data_hdm05.npz'
    database = np.load(os.path.join(data_path, db))['clips']

    for i in range(10):
        index0 = np.random.randint(0, len(database))
        index1 = np.random.randint(0, len(database))
        index2 = np.random.randint(0, len(database))

        animation_plot([
            database[index0:index0 + 1],
            database[index1:index1 + 1],
            database[index2:index2 + 1],
        ])
