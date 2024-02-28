import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import fonts

from uq.utils.general import plot_or_savefig

from .dataframes import build_grouped_comparison_df


def build_diff_df(df, metrics, baseline_query, join_by, columns_to_keep):
    grouped = build_grouped_comparison_df(df, metrics, baseline_query, join_by, columns_to_keep)
    df_diff = grouped.apply(
        lambda x: (x['compared'].astype(float).to_numpy() - x['baseline'].astype(float).to_numpy()).mean()
    )
    df_diff = df_diff.reset_index(name='diff')
    df_diff['metric'] = pd.Categorical(df_diff['metric'], metrics)
    df_diff = df_diff.sort_values('metric', kind='stable')
    return df_diff


def plot_regularization_impact_on_axis(axis, df_diff, metric):
    df_diff = df_diff['diff'].dropna()

    if metric == 'test_calib_l1':
        min_log, max_log, step = -3, 0, 0.5
        rlogxticks = -3, 0, 1
    elif metric == 'test_posthoc_calib_l1':
        min_log, max_log, step = -3, 0.5, 0.5
        rlogxticks = -3, 0, 1
    elif metric == 'test_crps':
        min_log, max_log, step = -1, 8, 2
        rlogxticks = -1, 7, 2
    elif metric == 'test_nll':
        min_log, max_log, step = -1, 3, 0.5
        rlogxticks = -1, 2, 1
    elif metric == 'test_stddev':
        min_log, max_log, step = -1, 8, 2
        rlogxticks = -1, 7, 2
    else:
        raise ValueError('Bins not specified')
    rbins = 10.0 ** np.arange(min_log, max_log, step)
    # To put again
    # assert metric_diff.abs().max() < rbins.max(), (metric_diff.abs().max(), rbins.max())

    bins = np.concatenate([-rbins[::-1], [0], rbins])
    rxticks = 10.0 ** np.arange(*rlogxticks)
    xticks = np.concatenate([-rxticks[::-1], [0], rxticks])
    axis.hist(df_diff, bins=bins)
    axis.set_xscale('symlog', linthresh=rbins.min(), linscale=step)
    axis.set_xticks(xticks)
    axis.axvline(x=0, color='red', ls='--')
    left_count = (df_diff.to_numpy() < 0).astype(int).sum()
    right_count = (df_diff.to_numpy() > 0).astype(int).sum()
    axis.text(
        0.4,
        0.8,
        str(left_count),
        transform=axis.transAxes,
        va='center',
        ha='center',
        size=14,
    )
    axis.text(
        0.6,
        0.8,
        str(right_count),
        transform=axis.transAxes,
        va='center',
        ha='center',
        size=14,
    )


def plot_regularization_impact_per_row(df, baseline_queries, row_queries, col_metrics):
    nrows, ncols = len(row_queries), len(col_metrics)
    size = nrows * ncols
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3, nrows * 1.5),
        squeeze=False,
        sharex='col',
        sharey='col',
        dpi=200,
    )

    for row, baseline_query, row_query in zip(range(nrows), baseline_queries, row_queries.values()):
        for col, metric in zip(range(ncols), col_metrics):
            axis = ax[row][col]
            # plot_df = df.query(f'metric == "{metric}"')
            join_by = ['dataset_group', 'dataset', 'metric']
            columns_to_keep = []
            df_diff = build_diff_df(
                df.query(row_query),
                [metric],
                baseline_query,
                join_by,
                columns_to_keep,
            )
            plot_regularization_impact_on_axis(axis, df_diff, metric)
            # axis.set(title=rf'\texttt{{{model_name}}}')
            axis.tick_params(axis='x', which='major', labelsize=8)
            axis.tick_params(axis='x', which='minor', labelsize=6)
            # axis.set(xlabel=f'Difference of {metric}', ylabel='Count')
    for row, query_name in zip(range(nrows), row_queries):
        ax[row, 0].set(ylabel=query_name)
        # ax[i, 0].set(ylabel='Count')
    for col, metric in zip(range(ncols), col_metrics):
        ax[-1, col].set(xlabel=rf'Difference of \texttt{{{metric}}}')
    fig.tight_layout()
    return fig


# TO CHANGE
def plot_regularization_impact_same_metric(
    df,
    metric,
    baseline_query,
    regul_queries,
    ncols=3,
    alpha=0.5,
    nb_hidden=3,
    mixture_size=3,
):
    size = len(regul_queries)
    nrows = math.ceil(size / ncols)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3, nrows * 1.5),
        squeeze=False,
        sharex=True,
        sharey=True,
        dpi=200,
    )
    ax_flatten = ax.flatten()
    for i in range(size, len(ax_flatten)):
        ax_flatten[i].set_visible(False)
    # fig, ax = plt.subplots(figsize=(8, 4), dpi=200, sharex=True, squeeze=False)
    for axis, (model_name, regul_query) in zip(ax_flatten, regul_queries.items()):
        plot_regularization_impact_on_axis(
            axis,
            df,
            metric,
            baseline_query,
            regul_query,
            alpha=alpha,
            nb_hidden=nb_hidden,
            mixture_size=mixture_size,
        )
        axis.set(title=rf'\texttt{{{model_name}}}')
        axis.tick_params(axis='x', which='major', labelsize=8)
        axis.tick_params(axis='x', which='minor', labelsize=6)
        # axis.set(xlabel=f'Difference of {metric}', ylabel='Count')
    for i in range(nrows):
        ax[i, 0].set(ylabel='Count')
    for i in range(ncols):
        ax[-1, i].set(xlabel=rf'Difference of \texttt{{{metric}}}')
    fig.tight_layout()
    path = Path('images') / 'hist_diff' / f'alpha_{alpha}' / f'mixture_size_{mixture_size}' / f'{metric}.png'
    plot_or_savefig(path, fig)


# TO CHANGE
def final_plot(test_df_mean, baseline_query, regul_queries):
    plot_df = test_df_mean.query('dataset_group != "toy"')
    with plt.rc_context(fonts.neurips2022_tex()):
        for metric in [
            'test_calib_l1',
            'test_posthoc_calib_l1',
        ]:   #'test_crps', 'test_stddev']:
            # for metric in ['test_stddev']:
            for alpha in [0.01, 0.1, 0.5, 0.9]:
                for mixture_size in [1, 3]:
                    plot_regularization_impact_same_metric(
                        plot_df,
                        metric,
                        baseline_query,
                        regul_queries,
                        alpha=alpha,
                        mixture_size=mixture_size,
                    )
