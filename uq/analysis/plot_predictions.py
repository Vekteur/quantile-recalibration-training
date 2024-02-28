import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from uq.metrics.calibration import quantile_calibration_from_pits_with_sorting
from uq.utils.general import savefig


def plot_marginal(axis, y, dist):
    ys = torch.linspace(-5, 5, 5000)
    with torch.no_grad():
        densities = dist.log_prob(ys[:, None]).exp()
    axis.plot(ys.numpy(), densities.flatten().numpy(), color='orange')
    axis.axvline(x=y, color='green', linestyle=':')
    axis.set_xlabel('y')
    axis.set_ylabel('Density')


def plot_predictions(model_name, model, posthoc_module, datamodule, nrows=5, ncols=6):
    n = nrows * ncols
    xs, ys = datamodule.data_test[:n]
    with torch.no_grad():
        dist = model.module.predict(xs)
        dist = posthoc_module.model(dist)

    alpha = torch.linspace(-5, 5,1000)
    densities = dist.log_prob(alpha[:, None]).exp().detach()
    nlls = -dist.log_prob(ys.squeeze(1))

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,
        sharex=True,
        dpi=200,
    )

    model.eval()
    for y, density, nll, axis in zip(ys.tolist(), densities.unbind(dim=1), nlls.tolist(), ax.flatten()):
        axis.plot(alpha, density, color='orange')
        axis.axvline(x=y, color='green', linestyle=':')
        axis.set_title(f'NLL: {nll:.3f}')

    for col in range(ncols):
        ax[-1, col].set_xlabel('y')
    for row in range(nrows):
        ax[row, 0].set_ylabel('density')

    fig.suptitle(f'Predictions of {model_name} on the dataset {datamodule.hparams.name}')
    fig.tight_layout()
    #fig.subplots_adjust(top=0.96)
    return fig


def plot_predictions_comparison_per_column(models_dict, datamodule, nrows=10):
    ncols = len(models_dict)
    n = nrows
    xs, ys = datamodule.data_test[10 : 10 + n]
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,
        sharex=True,
        dpi=200,
    )

    for col, (model_name, model) in enumerate(models_dict.items()):
        model.eval()
        with torch.no_grad():
            dist = model.module.predict(xs)
        quantiles = torch.linspace(-5, 5, 5000)
        densities = dist.log_prob(quantiles[:, None]).exp()
        nlls = -dist.log_prob(ys.squeeze(1))
        for y, density, nll, row in zip(ys.tolist(), densities.unbind(dim=1), nlls.tolist(), range(nrows)):
            axis = ax[row][col]
            axis.plot(quantiles, density, color='orange')
            axis.axvline(x=y, color='green', linestyle=':')
            axis.legend([], [], title=f'NLL: {nll:.3f}')

    for col in range(ncols):
        ax[-1, col].set_xlabel('y')
    for row in range(nrows):
        ax[row, 0].set_ylabel('density')

    for row in range(0, nrows, 5):
        for model_id, model_name in enumerate(models_dict.keys()):
            ax[row, model_id].set_title(model_name, fontsize=18, va='bottom')
    fig.tight_layout()
    return fig


def plot_predictions_comparison_sequentially(models_dict, datamodule, ncols=6):
    size = len(models_dict)
    nrows = math.ceil(size / ncols)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,
        sharex=True,
        dpi=200,
    )
    ax_flatten = ax.flatten()
    for i in range(size, len(ax_flatten)):
        ax_flatten[i].set_visible(False)

    n = 1
    xs, ys = datamodule.data_test[10 : 10 + n]

    for axis, (model_name, model) in zip(ax_flatten, models_dict.items()):
        model.eval()
        with torch.no_grad():
            dist = model.model.dist(xs)
        quantiles = torch.linspace(-5, 5, 5000)
        densities = dist.log_prob(quantiles[:, None]).exp()
        nlls = -dist.log_prob(ys.squeeze(1))
        for y, density, nll in zip(ys.tolist(), densities.unbind(dim=1), nlls.tolist()):
            axis.plot(quantiles, density, color='orange')
            axis.axvline(x=y, color='green', linestyle=':')
            axis.legend([], [], title=f'NLL: {nll:.3f}')
            axis.set_title(model_name, fontsize=18, va='bottom')

    for col in range(ncols):
        ax[-1, col].set_xlabel('y')
    for row in range(nrows):
        ax[row, 0].set_ylabel('density')
    fig.tight_layout()
    return fig


def plot_calibration(models_dict, datamodule, ncols=4):
    size = len(models_dict)
    nrows = math.ceil(size / ncols)
    xs, ys = datamodule.data_test[:]
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,
        sharex=True,
        sharey=True,
        dpi=200,
    )
    ax = ax.flatten()
    for i in range(size, len(ax)):
        ax[i].set_visible(False)

    for (model_name, model), axis in zip(models_dict.items(), ax):
        model.eval()
        with torch.no_grad():
            dist = model.model.dist(xs)
        pits = dist.cdf(ys.squeeze(1))
        sorted_pits, _ = torch.sort(pits)
        lin = torch.linspace(0, 1, len(sorted_pits))
        axis.plot(sorted_pits, lin)

        # alpha = torch.linspace(0, 1, 100)
        # indicator = pits[:, None] <= alpha
        # mc_estimation = indicator.float().mean(dim=0)
        # axis.plot(alpha, mc_estimation, label='MC estimation')

        axis.plot([0, 1], [0, 1], color='black', linestyle='--')
        axis.set_xlabel('Forecasted probability')
        axis.set_ylabel('Observed frequency')
        axis.set_title(model_name)

    fig.tight_layout()
    return fig


def plot_calibration_comparison(models_dict, datamodule):
    xs, ys = datamodule.data_test[:]
    fig, axis = plt.subplots(1, 1, figsize=(5, 5), dpi=300)

    cm = mpl.cm.get_cmap('magma')
    color_linear = np.linspace(1, 0, len(models_dict) + 2)[1:-1]
    color_values = [cm(l) for l in color_linear]

    for (model_name, model), color in zip(models_dict.items(), color_values):
        model.eval()
        with torch.no_grad():
            dist = model.model.dist(xs)
        pits = dist.cdf(ys.squeeze(1))

        sorted_pits, _ = torch.sort(pits)
        lin = torch.linspace(0, 1, len(sorted_pits))
        calib = quantile_calibration_from_pits_with_sorting(pits, L=1)
        # label=f'{model_name}\n({calib:.3f})
        axis.plot(sorted_pits, lin, color=color, label=f'{model_name}')

    axis.plot([0, 1], [0, 1], color='black', linestyle='--')
    axis.set_xlabel('Forecasted probability')
    axis.set_ylabel('Observed frequency')

    fig.legend(bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=0.0)

    fig.tight_layout()
    return fig


def plot_recalibration_map(module, recalibration_map, axis):
    x = torch.linspace(0, 1, 100)
    with torch.no_grad():
        y = recalibration_map(x)
    axis.plot(x.numpy(), y.numpy(), '-b')
    axis.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    # axis.set(adjustable='box', aspect='equal')
    axis.set(title='Recalibration map')


def plot_inhoc(module, x, y, batch_idx, stage):
    if stage != 'train':
        return
    print(module.current_epoch, batch_idx)

    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), dpi=200)

    base_prediction_module = module.module.get_name('base')
    if base_prediction_module is not None:
        with torch.no_grad():
            dist = base_prediction_module.predict(x[:1])
        plot_marginal(ax[0, 0], y[:1], dist)
        ax[0, 0].set_title('Base predictions')
        handles, labels = ax[0, 0].get_legend_handles_labels()

    inhoc_module = module.module.get_name('inhoc')
    inhoc_method = inhoc_module.inhoc_hparams['method']
    if inhoc_method is not None:
        recalibration_map = inhoc_module.inhoc_model.recalibration_map
        plot_recalibration_map(module, recalibration_map, ax[0, 1])
        with torch.no_grad():
            dist = module.module.predict(x[:1])
        plot_marginal(ax[0, 2], y[:1], dist)
        ax[0, 2].set_title('Predictions of recalibrated training')

    ax[1, 0].hist(y, bins=100, density=True, color='orange', alpha=0.5)
    ax[1, 0].set_title('Marginal y distribution')

    axis = ax[1, 1]
    metrics = module.posthoc_manager.get_module({'method': None}).collector.metrics
    for metric_name in ['val_nll', 'calib_nll']:
        items = metrics['per_epoch'][metric_name]
        axis.plot(items.keys(), items.values(), label=metric_name)
        axis.legend(loc='upper left')

    axis = ax[1, 2]
    for metric_name in ['train_calib_l1', 'val_calib_l1', 'calib_calib_l1']:
        items = metrics['per_epoch'][metric_name]
        axis.plot(items.keys(), items.values(), label=metric_name)
        axis.legend(loc='upper left')

    fig.suptitle(f'Epoch {module.current_epoch}, step {batch_idx}')
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    fig.tight_layout()
    path = (
        Path('notebooks')
        / 'recal_training'
        / 'prediction_per_epoch'
        / str(inhoc_method)
        / f'{module.current_epoch}_{batch_idx}.png'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig(path, fig)
    plt.close()


def plot_predictions_per_epoch(module, batch_idx, stage, nrows=6, ncols=6):
    if stage != 'train':
        return
    print(module.current_epoch, batch_idx)

    n = nrows * ncols
    xs, ys = module.trainer.datamodule.data_test[:n]
    alpha = torch.linspace(ys.min() - 0.5, ys.max() + 0.5, 1000)
    with torch.no_grad():
        dist = module.module.predict(xs)
        densities = dist.log_prob(alpha[:, None]).exp()
        nlls = -dist.log_prob(ys.squeeze(1))

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,
        sharex=True,
    )

    for y, density, nll, axis in zip(ys.tolist(), densities.unbind(dim=1), nlls.tolist(), ax.flatten()):
        axis.plot(alpha, density, color='orange')
        axis.axvline(x=y, color='green', linestyle=':')
        axis.set_title(f'NLL: {nll:.3f}')

    for col in range(ncols):
        ax[-1, col].set_xlabel('y')
    for row in range(nrows):
        ax[row, 0].set_ylabel('density')

    inhoc_method = module.module.get_name('inhoc').inhoc_hparams['method']
    fig.suptitle(f'Epoch {module.current_epoch}, step {batch_idx}')
    fig.tight_layout()
    path = (
        Path('notebooks')
        / 'recal_training'
        / 'predictions_per_epoch'
        / str(inhoc_method)
        / f'{module.current_epoch}_{batch_idx}.png'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig(path, fig)
    plt.close()
