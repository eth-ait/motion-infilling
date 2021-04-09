"""
Creates Figure 5 which compares bone length consistency.
"""
import numpy as np
import matplotlib.pyplot as plt
from tbase.skeleton import Skeleton


def _plot(ax, bone_idxs, bone_names):
    gt_values = [Skeleton.BONE_LENGTHS[i] * Skeleton.TO_CM for i in bone_idxs]

    all_bone_lengths = []
    labels = []
    for idx in range(len(bone_idxs)):
        all_bone_lengths.append(bone_lengths_vgg[:, bone_idxs[idx]])
        all_bone_lengths.append(bone_lengths_holden[:, bone_idxs[idx]])
        labels.append(bone_names[idx])
        labels.append(bone_names[idx])

    box1 = ax.boxplot(all_bone_lengths[::2], showfliers=False, labels=labels[::2],
                      positions=list(range(1, 1 + len(labels), 2)),
                      patch_artist=True, widths=0.35, boxprops=dict(facecolor="C0"))
    plt.setp(box1['medians'], color='black')

    box2 = ax.boxplot(all_bone_lengths[1::2], showfliers=False, labels=labels[1::2],
                      positions=list(range(2, 2 + len(labels), 2)), patch_artist=True, widths=0.35,
                      boxprops=dict(facecolor="C1"))
    plt.setp(box2['medians'], color='black')

    b = ax.plot([1, 3, 5], gt_values, 'o', markersize=7, markerfacecolor='lightgray', markeredgecolor='black',
                markeredgewidth=0.5, zorder=10)
    ax.plot([2, 4, 6], gt_values, 'o', markersize=7, markerfacecolor='lightgray', markeredgecolor='black',
            markeredgewidth=0.5, zorder=10)

    ax.legend([box1["boxes"][0], box2["boxes"][0], b[0]], ['Ours', 'Holden et al.', 'Ground-Truth'], loc='upper left',
              prop={'size': 15})

    ax.set_xlim([0, len(labels) + 1])
    x_ticks = [x + 0.5 for x in range(1, len(labels) + 1, 2)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(bone_names)
    ax.set_ylabel('Bone Length [cm]')
    ax.yaxis.label.set_fontsize(20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)


if __name__ == '__main__':
    bone_lengths_vgg = np.load("..\\pretrained-models\\VGG\\run_031\\bone_lengths.npz")['bone_lengths']
    bone_lengths_holden = np.load("..\\pretrained-models\\HoldenCAE\\run_003\\bone_lengths.npz")['bone_lengths']

    fig, axs = plt.subplots(1, 2)
    _plot(axs[0], bone_idxs=[1, 2, 5], bone_names=['R Thigh', 'R Shin', 'L Thigh'])
    _plot(axs[1], bone_idxs=[3, 8, 14], bone_names=['R Foot', 'Spine', 'R Forearm'])

    print('n = ', bone_lengths_holden.shape[0])

    plt.show()
