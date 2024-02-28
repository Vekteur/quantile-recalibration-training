import logging
import math
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from tueplots import fonts

from .constants import metric_names
from .dataframes import build_grouped_comparison_df, make_df_abb
from .stats import cohen_d
from .cmaps import get_cmap

log = logging.getLogger('uq')


def debug_grouped_size(grouped):
    def agg(x):
        if len(x) > 5:
            display(x)

    grouped.apply(agg)


def build_cohen_d(df, metrics, baseline_query, join_by, scale=True):
    if len(df.query(baseline_query)) == 0:
        raise ValueError('No baseline found')
    grouped = build_grouped_comparison_df(df, metrics, baseline_query, join_by)
    sizes = grouped.size().value_counts().to_dict()
    print(
        'Size of groups:',
        ', '.join([f'{count} of size {size}' for size, count in sizes.items()]),
    )
    # debug_grouped_size(grouped)
    for size in sizes:
        assert size <= 5, size
    df_cohen = grouped.apply(
        lambda x: cohen_d(
            x['compared'].astype(float).to_numpy(),
            x['baseline'].astype(float).to_numpy(),
            scale=scale,
        )
    )
    if df_cohen.isna().all():
        print('WARNING: all cohen d are nan')
    name = "Cohen's d" if scale else 'Difference'
    df_cohen = df_cohen.rename(name).reset_index()
    df_cohen['metric'] = pd.Categorical(df_cohen['metric'], metrics)
    df_cohen = df_cohen.sort_values('metric', kind='stable')
    return df_cohen


def symmetrize_x_axis(axis):
    x_lim = np.abs(axis.get_xlim()).max()
    axis.set_xlim(xmin=-x_lim, xmax=x_lim)


def plot_cohen_d_barplot(df, col_queries, legend=True, figsize=None):
    n = len(col_queries)
    nrows, ncols = 1, n
    names = df.name.unique()
    if figsize is None:
        figsize = (5 * ncols, 0.3 * len(names) * nrows)
    fig, axes = plt.subplots(
        1,
        n,
        sharex=False,
        sharey=True,
        squeeze=False,
        figsize=figsize,
        dpi=300,
    )

    for axis, (name, query), i in zip(axes.flatten(), col_queries.items(), range(n)):
        # print(query, len(df.query(query)))
        data = df.query(query)
        if data["Cohen's d"].isna().all():
            continue
        model_name = data.apply(lambda d: f'{d["base_loss"]}\n({d["pred_type"]})', axis='columns')
        # x = data['base_loss'].astype(str) + ', ' + data['pred_type'].astype(str)
        axis.axvline(x=0, color='black', ls='--', zorder=1)
        g = sns.barplot(
            data,
            x="Cohen's d",
            y=model_name,
            hue='model',
            orient='h',
            errorbar=('se', 1),
            capsize=0.1,
            errwidth=1,
            ax=axis,
        )
        symmetrize_x_axis(axis)
        g.legend_.remove()
        # g.set_xticklabels(g.get_xticklabels(), rotation=10)
        # axis.set_title(name)
        axis.set_xlabel(f"Cohen's d of\n{name}", fontsize=9)
        axis.tick_params(axis='both', which='major', labelsize=9)
        # axis.set_ylabel('Base loss')

    if legend:
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            # title='Regularization',
            loc='lower center',
            bbox_to_anchor=(0.5, 1 - 0.05),
            frameon=True,
            ncol=4,
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def plot_sorted_boxplot(df, x=None, y=None, *args, **kwargs):
    medians = df.groupby(y)[x].median().sort_values()
    return sns.boxenplot(df, x=x, y=y, order=medians.index, *args, **kwargs)





def plot_cohen_d_boxplot(df, metrics, legend=False, figsize=None, cmap=None, ncols=4, wspace=0.35):
    col_queries = {
        metric: f'metric == "{metric}"'
        for metric in metrics
    }
    col_queries = {name: query for name, query in col_queries.items() if len(df.query(query)) > 0}
    size = len(col_queries)
    nrows = math.ceil(size / ncols)
    names = df.name.unique()
    if figsize is None:
        figsize = (2 * ncols, 0.8 + 0.15 * len(names) * nrows)
        #figsize = (3.1 * ncols, 1.6 + 0.15 * len(names) * nrows)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=figsize,
        dpi=300,
    )
    ax_flatten = ax.flatten()
    for i in range(size, len(ax_flatten)):
        ax_flatten[i].set_visible(False)

    cmap = get_cmap(df, cmap)

    for axis, (name, query) in zip(ax_flatten, col_queries.items()):
        # print(query, len(df.query(query)))
        data = df.query(query)
        if data["Cohen's d"].isna().all():
            continue
        # model_name = data.apply(lambda d: f'{d["base_loss"]}\n({d["pred_type"]})', axis='columns')
        # x = data['base_loss'].astype(str) + ', ' + data['pred_type'].astype(str)
        g = plot_sorted_boxplot(
            data,
            x="Cohen's d",
            y='name',
            orient='h',
            linewidth=1,
            palette=cmap,
            hue='name',
            legend=False,
            ax=axis,
            flier_kws={'s': 2},
        )
        #axis.axvline(x=0, color='black', ls=(0, (3, 3)), lw=1, zorder=1)

        cp = ConnectionPatch(
            xyA=(0, -0.1),
            xyB=(0, 1.1),
            coordsA=axis.get_xaxis_transform(),
            coordsB=axis.get_xaxis_transform(),
            axesA=axis,
            axesB=axis,
        )
        fig.add_artist(cp)
        for y in [-0.1, 1.1]:
            cp = ConnectionPatch(
                xyA=(-0.1, y),
                xyB=(0.1, y),
                coordsA=axis.get_xaxis_transform(),
                coordsB=axis.get_xaxis_transform(),
                axesA=axis,
                axesB=axis,
            )
            fig.add_artist(cp)

        # offset = mpl.transforms.ScaledTranslation(0, -5/72., fig.dpi_scale_trans)
        # for label in axis.xaxis.get_majorticklabels():
        #     # We only offset the label at position 0
        #     if label.get_position()[0] == 0:
        #         label.set_transform(label.get_transform() + offset)


        g.set_xscale('symlog')
        symmetrize_x_axis(axis)
        # g.legend_.remove()
        # g.set_xticklabels(g.get_xticklabels(), rotation=10)
        # axis.set_title(name)
        metric_name = metric_names[name.split('_', 1)[-1]]
        axis.set_xlabel(f"Cohen's d of {metric_name}", fontsize=9)
        axis.set_ylabel('')
        axis.tick_params(axis='both', which='major', labelsize=7)
        # axis.set_ylabel('Base loss')

    if legend:
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            # title='Regularization',
            loc='lower center',
            bbox_to_anchor=(0.5, 1 - 0.05),
            frameon=True,
            ncol=4,
            fontsize=9,
        )
    fig.tight_layout()
    fig.subplots_adjust(wspace=wspace)
    return fig


def series_to_int(series):
    d = {value: order for order, value in enumerate(series.unique())}
    return series.map(d)


def plot_cohen_d_indexed(df):
    df = df.copy()
    df['dataset'] = series_to_int(df['dataset'])
    df = df.sort_values('dataset', kind='stable')
    metrics = df.metric.unique()
    n = len(metrics)
    nrows, ncols = n, 1
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        squeeze=False,
        figsize=(4 * ncols, 2 * nrows),
        dpi=300,
    )
    for axis, (metric, df_metric), i in zip(axes.flatten(), df.groupby('metric'), range(n)):
        g = sns.scatterplot(
            df_metric,
            x='dataset',
            y="Cohen's d",
            hue='model',
            style='pred_type',
            legend='full',
            ax=axis,
        )
        if i != n - 1:
            g.set(xlabel=None)
        axis.set_title(metric)

    fig.tight_layout()
    return fig


def build_diff_df(df, metrics, baseline_query, join_by):
    merged = build_grouped_comparison_df(df, metrics, baseline_query, join_by, group=False)
    df_diff = merged['compared'] - merged['baseline']
    df_diff = df_diff.rename('Difference').reset_index()
    return df_diff



def plot_metric_difference(df_diff, metric, cmap=None, mixture_size=[3], base_model='nn', name='some_difference'):
    cmap = get_cmap(df_diff, cmap)
    names = list(cmap.keys())
    df_diff['name'] = pd.Categorical(df_diff['name'], names)
    df_diff = df_diff.sort_values('name')
    df_abb = make_df_abb(df_diff.reset_index()['dataset'].unique())
    df_diff = df_diff.merge(df_abb, on='dataset')

    # Order by Difference
    name_for_order = names[0]
    selected = df_diff.query('metric == @metric and name == @name_for_order')[['dataset', 'Difference']]
    selected = selected.groupby('dataset', as_index=False).mean()
    datasets_order = selected.sort_values('Difference').dataset
    order_map = {dataset: i for i, dataset in enumerate(datasets_order)}
    df_diff = df_diff.sort_values('dataset', key=lambda x: x.map(order_map), kind='stable')

    fig, axis = plt.subplots(figsize=(18, 3.4))
    axis.grid(axis='y')
    axis.set_axisbelow(True)
    #df_diff = latex_names(df_diff)
    g = sns.barplot(df_diff, 
        x='abb', y='Difference', hue='name', 
        orient='v',
        #errorbar=None,
        errorbar=('se', 1),
        capsize=0.1,
        err_kws={'linewidth': 1},
        ax=axis,
        palette=cmap,
    )
    g.legend().set_title(None)
    plt.setp(axis.get_legend().get_texts(), fontsize=15)
    axis.set_yscale('symlog', linthresh=1e-1)
    axis.tick_params(axis='x', which='major', labelrotation=90, labelsize=15)
    metric_name = metric_names[metric.split('_', 1)[-1]]
    axis.set_ylabel(f'Test {metric_name} difference', fontsize=20)
    axis.set_xlabel(None)
    axis.axhline(y=0, color='black', ls='--', zorder=1)
    return fig
