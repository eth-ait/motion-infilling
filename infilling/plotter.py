import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable


def _scale_foot_contacts(img, low=0.0, high=1.0):
    """
    Scale the foot contacts which are originally are in range [`low`, `high`] to [`min, max`] of whole image to make
    them better visible.
    
    :param img: A 2-dimensional np-array.
    :param low: the lower boundary of the original range of the foot contacts.
    :param high: the upper boundary of the original range of the foot contacts.
    """
    assert len(img.shape) == 2
    assert (high - low ) > 0.0

    img_c = np.copy(img)
    actual_min = np.amin(img_c)
    actual_max = np.amax(img_c)

    # transform to interval [0, 1]
    div = high - low
    x_std = (img_c[-4:] - low) / div

    # transform to interval [actual_min, actual_max]
    x_norm = x_std * (actual_max - actual_min) + actual_min
    img_c[-4:] = x_norm
    return img_c


def _squeeze(array):
    sq = np.squeeze(array, axis=0) if array.shape[0] == 1 else array
    sq = np.squeeze(sq, axis=-1) if sq.shape[-1] == 1 and len(sq.shape) == 3 else sq
    assert len(sq.shape) == 2, 'unexpected squeezing'
    return sq


def plot_reconstruction(input_, target, mask, reconstructions, title,
                        scale_foot_contacts_to=None, sub_titles=None, save_path=None, show=False):
    """
    Plot original input, target and reconstructed data into a figure. Input arrays can either be 3 dimensional, where
    the first dimension has size 1 or regular 2-dimensional images.

    :param input_: A 3- or 2-dimensional np-array,
    :param target: A 3- or 2-dimensional np-array,
    :param reconstructions: A list of 3- or 2-dimensional np-array. Each entry will be printed on a new row showing
      the reconstruction and the difference to the original.
    :param mask: The mask used on the input.
    :param title: A title for the figure.
    :param scale_foot_contacts_to: If set, scales the foot contacts to the 2 values given as a tuple.
    :param sub_titles: Titles for each row showing the reconstruction.
    :param save_path: Path to where the input arrays and figure should be saved to or None if nothing should be saved
    :param show: If set, the figure will be shown on screen.
    """
    reconstructions = [reconstructions] if not isinstance(reconstructions, list) else reconstructions
    sub_titles = [str(i) for i in range(len(reconstructions))] if sub_titles is None else sub_titles
    assert len(sub_titles) == len(reconstructions)
    assert scale_foot_contacts_to is None or len(scale_foot_contacts_to) == 2

    def _scale_foot_contacts_if_required(img):
        if scale_foot_contacts_to is not None and img.shape[1] > 69:
            scaled = _scale_foot_contacts(img,
                                          low=scale_foot_contacts_to[0],
                                          high=scale_foot_contacts_to[1])
        else:
            scaled = img
        return scaled

    tar_ori = _squeeze(target)
    in_ori = _squeeze(input_)
    tar_scaled = _scale_foot_contacts_if_required(tar_ori)
    in_scaled = _scale_foot_contacts_if_required(in_ori)

    nrows = len(reconstructions) + 1
    height = 3 * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, height))
    fig.suptitle(title)

    ax0 = axes[0, 0]
    ax1 = axes[0, 1]

    plt.rcParams['image.cmap'] = 'viridis'

    # plot (masked) input and target
    im = ax0.imshow(np.ma.array(in_scaled, mask=mask))
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax0.set_title('input')

    im = ax1.imshow(tar_scaled)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax1.set_title('original')

    # before plotting the reconstructions compute all diffs so that we can scale everything to min and max error
    diffs = []
    recs_scaled = []
    for i in range(1, nrows):
        rec_ori = _squeeze(reconstructions[i - 1])
        rec_scaled = _scale_foot_contacts_if_required(rec_ori)
        recs_scaled.append(rec_scaled)
        diffs.append(np.fabs(tar_ori - rec_ori))

    # now compute min and max for all diffs and map into interval [0, 1]
    x_min = np.amin([np.amin(x) for x in diffs])
    x_max = np.amax([np.amax(x) for x in diffs])
    x_div = x_max - x_min
    diffs_scaled = [(d - x_min) / x_div for d in diffs]

    # plot the reconstructions
    for i in range(1, nrows):
        ax2 = axes[i, 0]
        ax3 = axes[i, 1]

        im = ax2.imshow(recs_scaled[i-1])
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax2.set_title('reconstructed ({})'.format(sub_titles[i-1]))

        im = ax3.imshow(diffs_scaled[i-1])
        im.set_cmap('jet')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax3.set_title('difference to original')

    fig.tight_layout()

    if show:
        plt.show()

    # save figure and data to disk
    if save_path is not None:
        recs = np.vstack(reconstructions)
        fig.savefig('{}.png'.format(save_path), dpi=150)
        np.savez_compressed('{}.npz'.format(save_path), ori=tar_ori, rec=recs, ori_in=_squeeze(input_))

    plt.close(fig)


def show_np_arrays(data, title, subtitles):
    assert len(data) == len(subtitles)

    fig, axes = plt.subplots(nrows=len(data), ncols=1)
    fig.suptitle(title)

    for i, x in enumerate(data):
        x = _squeeze(x)
        ax = axes[i] if len(data) > 1 else axes

        im = ax.imshow(x)
        im.set_cmap('viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(subtitles[i])

    plt.show()
