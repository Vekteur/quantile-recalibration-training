import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from uq.analysis.plot_predictions import plot_recalibration_map
from uq.utils.general import savefig


def get_prediction_domain(x_test, num_y_test, y_train):
    left, right = x_test.min(), x_test.max()
    max_dist = y_train.max() - y_train.min()
    padding = max_dist / 5
    bottom, top = y_train.min() - padding, y_train.max() + padding

    y_test = torch.linspace(bottom, top, num_y_test)
    extent = (left.item(), right.item(), bottom.item(), top.item())

    return y_test, extent


def plot_pdf(axis, pdf, extent):
    colors_list = [(1, 1, 1, 0), (1, 0.6, 0.15, 1)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom', colors_list, N=10000)
    return axis.imshow(
        pdf,
        interpolation='bilinear',
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        norm=mpl.colors.LogNorm(vmin=0.01, vmax=5),
    )


def plot_toy_dataset(axis, x_train, y_train, x_test, dist_test, name=None, scatter_alpha=0.5):
    num_y_test = 500
    y_test, extent = get_prediction_domain(x_test, num_y_test, y_train)
    pdf = dist_test.log_prob(y_test[:, None]).exp().detach().numpy()
    im = plot_pdf(axis, pdf, extent)

    # Plot the training data
    axis.scatter(
        x_train.numpy(),
        y_train.numpy(),
        s=2,
        alpha=scatter_alpha,
        label=f'Training points ({y_train.shape[0]})',
    )

    axis.set_xlim(extent[0], extent[1])
    axis.grid(True)
    axis.set_xlabel('$x$')
    axis.set_ylabel('$y$', rotation=0)
    if name is not None:
        axis.set_title(name.strip())
    return im


def plot_toy_dataset_with_color_bar(fig, axis, *args, **kwargs):
    im = plot_toy_dataset(axis, *args, **kwargs)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.02)
    fig.colorbar(im, cax=cax, orientation='vertical')


class ToyGraphics:
    def __init__(self, size, ncols=3):
        self.ncols = min(ncols, size)
        self.nrows = math.ceil(size / ncols)
        self.fig, self.axes = plt.subplots(
            self.nrows,
            self.ncols,
            squeeze=False,
            figsize=(self.ncols * 4, self.nrows * 3),
            dpi=200,
            sharex=True,
            sharey=True,
        )
        self.axes = self.axes.flatten()
        for i in range(size, len(self.axes)):
            self.axes[i].set_visible(False)
        self.axis_id = 0

    def plot(self, *args, **kwargs):
        plot_toy_dataset_with_color_bar(self.fig, self.axes[self.axis_id], *args, **kwargs)
        self.axis_id += 1

    def save(self, filepath, title=None):
        handles, labels = self.axes[0].get_legend_handles_labels()
        ncol_legend = self.ncols
        self.fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0),
            frameon=True,
            ncol=ncol_legend,
        )
        self.fig.suptitle(title)
        self.fig.tight_layout(rect=[0, 0.15 / self.nrows, 1, 1])
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(filepath)


def get_true_dist(module, x_test):
    import torch.distributions as D

    from uq.datamodules.toy.toy_module import get_toy_cond_dist

    x_test = module.trainer.datamodule.scaler_x.inverse_transform(x_test)
    prefix = module.trainer.datamodule.hparams.name.rsplit('_', 1)[0]
    dist = get_toy_cond_dist(prefix)(x_test.flatten())
    dist = D.TransformedDistribution(dist, D.AffineTransform(module.scaler.mean_, module.scaler.scale_).inv)
    return dist


def plot_inhoc(module, x, y, batch_idx, stage, legend=False):
    if stage != 'train':
        return
    print(module.current_epoch, batch_idx)

    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), dpi=200)

    def get_x_test():
        x_test, y_test = module.trainer.datamodule.data_test[:]
        num_x = 500
        return torch.linspace(x_test.min(), x_test.max(), num_x).unsqueeze(dim=-1)

    x_test = get_x_test()

    base_prediction_module = module.module.get_name('base')
    if base_prediction_module is not None:
        with torch.no_grad():
            dist = base_prediction_module.predict(x_test)
        plot_toy_dataset_with_color_bar(fig, ax[0, 0], x, y, x_test, dist)
        ax[0, 0].set_title('Base predictions')
        handles, labels = ax[0, 0].get_legend_handles_labels()

    inhoc_module = module.module.get_name('inhoc')
    inhoc_method = inhoc_module.inhoc_hparams['method']
    if inhoc_method is not None:
        recalibration_map = inhoc_module.inhoc_model.recalibration_map
        plot_recalibration_map(module, recalibration_map, ax[0, 1])
        with torch.no_grad():
            dist = module.module.predict(x_test)
        plot_toy_dataset_with_color_bar(fig, ax[0, 2], x, y, x_test, dist)
        ax[0, 2].set_title('Predictions of recalibrated training')

    dist = get_true_dist(module, x_test)
    plot_toy_dataset_with_color_bar(fig, ax[1, 0], x, y, x_test, dist)
    ax[1, 0].set_title('True distribution (unknown)')

    axis = ax[1, 1]
    metrics = module.posthoc_manager.get_module({'method': None}).collector.metrics
    for metric_name in ['train_nll', 'val_nll', 'calib_nll']:
        items = metrics['per_epoch'][metric_name]
        axis.plot(items.keys(), items.values(), label=metric_name)
        axis.legend(loc='upper left')

    axis = ax[1, 2]
    for metric_name in ['train_calib_l1', 'val_calib_l1', 'calib_calib_l1']:
        items = metrics['per_epoch'][metric_name]
        axis.plot(items.keys(), items.values(), label=metric_name)
        axis.legend(loc='upper left')

    if legend:
        patch = mpl.patches.Patch(color='orange', label='Density')
        handles += [patch]
        labels += ['Density']
        fig.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            frameon=True,
            ncol=2,
            fontsize=14,
            title_fontsize=14,
        )

    fig.suptitle(f'Epoch {module.current_epoch}, step {batch_idx}')
    fig.tight_layout()
    path = (
        Path('notebooks')
        / 'recal_training'
        / 'pdf_per_epoch'
        / str(inhoc_method)
        / f'{module.current_epoch}_{batch_idx}.png'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig(path, fig)
    plt.close()

def make_gif(images_dir, gif_path, ext='png'):
    import imageio.v3 as iio
    from natsort import natsorted
    from pygifsicle import optimize

    images_paths = natsorted(images_dir.rglob(f'*.{ext}'))
    frames = np.stack([iio.imread(path, extension=ext) for path in images_paths], axis=0)
    iio.imwrite(gif_path, frames, duration=300)
    optimize(gif_path)
