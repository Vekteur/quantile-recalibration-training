import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from .dataframes import get_all_metrics
from .cmaps import get_cmap


def plot_unit(x, y, path, name, unit='epoch'):
    # Using `fig, axis = plt.subplots()` results in a memory leak
    fig = mpl.figure.Figure()
    axis = fig.subplots()
    axis.plot(x, y)
    axis.set(xlabel=unit, ylabel=name, title=f'{name} per {unit} during training')
    fig.savefig(path / f'{name}.png')
    plt.close(fig)


def add_train_val_metrics(df, train_val_metrics=None):
    df = df.copy()
    if train_val_metrics is None:
        train_val_metrics = {
            metric
            for metric in get_all_metrics(df)
            if (metric.startswith('train_') or metric.startswith('val_'))
            and not (metric.startswith('train_quantile_score_') or metric.startswith('val_quantile_score_'))
        }
    columns_to_concat = []
    for metric in train_val_metrics:
        columns_to_concat.append(df.metrics.map(lambda d: d['per_epoch'][metric]).rename(metric))
    df = pd.concat([df] + columns_to_concat, axis=1)
    df = df.copy()   # copy is used to defragment the dataframe
    return df


def make_df_per_metric_and_epoch(df, metrics=None):
    df = df.set_index(['dataset_group', 'dataset', 'name', 'run_id'])
    df = df[list(metrics)].stack()
    df = df.rename_axis(index={None: 'metric'})
    df = df.apply(lambda x: pd.Series(x, dtype=float)).stack(dropna=False)
    df = df.rename_axis(index={None: 'epoch'})
    df = df.to_frame(name='value').reset_index()
    df = df.sort_values('metric', kind='stable')
    return df


def make_plot_df(df, train_val_metrics=None):
    """
    Transform the dataframe to a dataframe with columns
    ('dataset_group', 'dataset', 'name', 'run_id', 'metric', 'epoch')
    """
    df = add_train_val_metrics(df, train_val_metrics=train_val_metrics)
    return make_df_per_metric_and_epoch(df, train_val_metrics)


# Doing it with matplotlib is more easily customizable and a lot faster
def plot_dataset_seaborn(df):
    df_plot = make_plot_df(df)
    g = sns.FacetGrid(
        df_plot,
        row='metric',
        col='name',
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    g.map(sns.lineplot, 'epoch', 'value', ci=90)


def sem(x):
    return scipy.stats.sem(x, ddof=0)


def plot_sem(axis, df, label=None, color=None):
    df = df.query('not mean.isna()')
    axis.plot(df.index, df['mean'], label=label, color=color)
    axis.fill_between(
        df.index,
        df['mean'] - df['sem'],
        df['mean'] + df['sem'],
        alpha=0.2,
        color=color,
        zorder=10,
    )


def plot_all_runs(axis, df, label=None, color=None):
    for run_id, df_run in df.groupby('run_id'):
        label = label if run_id == 0 else None
        axis.plot(df_run['epoch'], df_run['value'], label=label, color=color, lw=0.5)


def plot_agg_runs(axis, df, label=None, color=None):
    df = df.drop(columns='run_id').groupby('epoch').agg(['mean', sem])
    df = df.droplevel(0, axis=1)   # Remove the 'value' level
    plot_sem(axis, df, label=label, color=color)


def plot_runs(*args, agg_run=True, **kwargs):
    if agg_run:
        plot_agg_runs(*args, **kwargs)
    else:
        plot_all_runs(*args, **kwargs)


from IPython.display import display


def plot_hline(axis, df, df_best_iter, color=None, agg_run=True):
    df_best_iter = df_best_iter.reset_index()[['run_id', 'epoch']]
    df_merged = pd.merge(df_best_iter, df)
    ys = df_merged.value
    if agg_run:
        ys = [ys.mean()]
    for y in ys:
        axis.axhline(y, color=color, linestyle='--', lw=1, zorder=100)


def plot_vline(axis, df_best_iter, color=None, agg_run=True):
    if agg_run:
        df_best_iter = [df_best_iter.mean()]
    for iter in df_best_iter:
        axis.axvline(iter, color=color, linestyle='--', lw=1, zorder=100)
    

def plot_posthoc(axis, df, df_best_iter, label=None, color=None, agg_run=True):
    def get_xs(df_best_iter):
        if agg_run:
            df_best_iter = [df_best_iter.mean()]
        return df_best_iter

    def get_ys(df_best_iter):
        df_best_iter = df_best_iter.reset_index()[['run_id', 'epoch']]
        df_merged = pd.merge(df_best_iter, df)
        ys = df_merged.value
        if agg_run:
            ys = [ys.mean()]
        return ys

    xs, ys = get_xs(df_best_iter), get_ys(df_best_iter)
    for i, (x, y) in enumerate(zip(xs, ys)):
        label = label if i == 0 else None
        axis.scatter(x, y, color=color, label=label, marker='*', edgecolors='black', lw=0.3, s=160, zorder=100)



def plot_runs_and_lines(axis, df_plot, df_best_iter, label=None, color=None, agg_run=True, posthoc=False):
    if posthoc:
        plot_posthoc(axis, df_plot, df_best_iter, label=label, color=color, agg_run=agg_run)
    else:
        plot_runs(axis, df_plot, label=label, color=color, agg_run=agg_run)
        plot_vline(axis, df_best_iter, color=color, agg_run=agg_run)
        plot_hline(axis, df_plot, df_best_iter, color=color, agg_run=agg_run)


def get_best_iter(df, model_name):
    df = df.query(f'metric == "val_es_loss" and name == @model_name')
    df_best_iter = df.loc[df.groupby(['dataset', 'run_id'])['value'].idxmin()]
    res = df_best_iter.set_index(['dataset', 'name', 'run_id', 'metric']).epoch
    return res


# Ad hoc function
# def get_colors(cmap, models):
#     if set(models).issubset({'BASE', 'QRC', 'QRT', 'QRTC'}):
#         return {
#             'BASE': 'tab:blue',
#             'QRC': 'tab:blue',
#             'QRT': 'tab:green',
#             'QRTC': 'tab:green',
#         }

#     colors = mpl.colors.TABLEAU_COLORS
#     assert len(models) <= len(colors)
#     return {
#         model: color
#         for model, color in zip(models, colors)
#     }


def plot_metric_comparison_per_epoch(
    df,
    facets,
    facets_order=None,
    agg_run=True,
    cmap=None,
    ncols=3,
    ncols_legend=3,
    sharex=False,
):
    df = df.sort_values('epoch', kind='stable')
    plot_both = agg_run == 'both'
    if plot_both:
        agg_run = False

    models = df['name'].unique()
    cmap = get_cmap(df, cmap)

    if facets_order is None:
        facets_order = df[facets].unique()
    size = len(facets_order)
    nrows = math.ceil(size / ncols)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.2, nrows * 2.5),
        squeeze=False,
        sharex=sharex,
        dpi=200,
    )
    ax = ax.flatten()
    for i in range(size, len(ax)):
        ax[i].set_visible(False)

    model_to_best_iter = {model_name: get_best_iter(df, model_name) for model_name in models}

    df = df.query(f'metric != "val_es_loss"')

    for facet_id, facet in enumerate(facets_order):
        df_facet = df.query(f'{facets} == "{facet}"')
        unique_metrics = df_facet['metric'].unique()
        assert len(unique_metrics) == 1, f'There should be only one metric per facet ({unique_metrics})'
        (metric,) = df_facet['metric'].unique()
        axis = ax[facet_id]
        for model_id, model_name in enumerate(models):
            df_plot = df_facet.query('name == @model_name')
            df_best_iter = model_to_best_iter[model_name]
            index = df_best_iter.index.names
            # df_best_iter = df_best_iter.reset_index().query(f'{facets} == @facet').set_index(index)['value']
            df_best_iter = df_best_iter.loc[df_best_iter.index.get_level_values(facets) == facet]
            df_plot = df_plot[['run_id', 'epoch', 'value']]  # .drop(columns=['metric', 'name'])
            if not df_plot.empty:
                posthoc = model_name.endswith('C') or model_name == r'\textbf{QRTC}'
                plot_runs_and_lines(
                    axis,
                    df_plot,
                    df_best_iter,
                    label=model_name,
                    color=cmap[model_name],
                    agg_run=agg_run,
                    posthoc=posthoc,
                )
                if plot_both:
                    plot_runs_and_lines(
                        axis,
                        df_plot,
                        df_best_iter,
                        label=None,
                        color='black',
                        agg_run=not agg_run,
                    )
        axis.set_title(facet, fontsize=17)
        axis.set_xlabel('Epoch', fontsize=14)
        axis.margins(x=0.02)
        if '_calib_l' in metric or '_length_' in metric or '_stddev' in metric:
            axis.set_ylim(bottom=0)
        if facet_id == 0:
            handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        frameon=True,
        ncol=ncols_legend,
        fontsize=20,
        title_fontsize=14,
    )
    fig.tight_layout()
    # fig.subplots_adjust(top=0.96)
    return fig


def plot_metric_comparison_per_dataset_per_epoch(
    df,
    datasets,
    metrics,
    agg_run=True,
    cmap=None,
    ncols_legend=3,
    sharex='row',
):
    df = df.sort_values('epoch', kind='stable')
    plot_both = agg_run == 'both'
    if plot_both:
        agg_run = False

    models = df['name'].unique()
    cmap = get_cmap(df, cmap)

    nrows = len(datasets)
    ncols = len(metrics)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.5, nrows * 2.5),
        squeeze=False,
        sharex=sharex,
        dpi=200,
    )

    model_to_best_iter = {model_name: get_best_iter(df, model_name) for model_name in models}

    df = df.query(f'metric != "val_es_loss"')

    for dataset_id, dataset in enumerate(datasets):
        for metric_id, metric in enumerate(metrics):
            df_facet = df.query(f'dataset == "{dataset}"')
            df_facet = df_facet[df_facet['metric'] == metric]
            unique_metrics = df_facet['metric'].unique()
            if len(unique_metrics) == 0:
                print(f'No metric for {metric} on {dataset}')
                continue
            assert len(unique_metrics) == 1, f'There should be only one metric per facet ({unique_metrics})'
            (metric,) = df_facet['metric'].unique()
            axis = ax[dataset_id][metric_id]
            for model_id, model_name in enumerate(models):
                df_plot = df_facet.query('name == @model_name')
                df_best_iter = model_to_best_iter[model_name]
                df_best_iter = df_best_iter.loc[df_best_iter.index.get_level_values('dataset') == dataset]
                df_plot = df_plot[['run_id', 'epoch', 'value']]  # .drop(columns=['metric', 'name'])
                if not df_plot.empty:
                    posthoc = model_name.endswith('C') or model_name == r'\textbf{QRTC}'
                    plot_runs_and_lines(
                        axis,
                        df_plot,
                        df_best_iter,
                        label=model_name,
                        color=cmap[model_name],
                        agg_run=agg_run,
                        posthoc=posthoc,
                    )
                    if plot_both:
                        plot_runs_and_lines(
                            axis,
                            df_plot,
                            df_best_iter,
                            label=None,
                            color='black',
                            agg_run=not agg_run,
                        )
            axis.margins(x=0.02)
            if '_calib_l' in metric or '_length_' in metric or '_stddev' in metric:
                axis.set_ylim(bottom=0)
            if metric_id == 0:
                handles, labels = axis.get_legend_handles_labels()

    for dataset_id, dataset in enumerate(datasets):
        dataset = f'Metric value on\n dataset {dataset}'
        ax[dataset_id, 0].set_ylabel(dataset, fontsize=20, rotation=90, va='bottom')
    fig.align_ylabels(ax[:, 0])
    for metric_id, metric in enumerate(metrics):
        ax[0, metric_id].set_title(metric, fontsize=20, va='bottom')
        ax[-1, metric_id].set_xlabel('Epoch', fontsize=14)

    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        frameon=True,
        ncol=ncols_legend,
        fontsize=20,
    )
    fig.tight_layout()
    return fig


def plot_metric_comparison_per_epoch_with_models_as_columns(df, agg_run=True, filter_runs=None):
    fixed_columns = ['dataset_group', 'dataset']
    for col in fixed_columns:
        assert df[col].nunique() == 1
    df = df.drop(columns=fixed_columns)

    if filter_runs is not None:
        df = df[df['run_id'].isin(filter_runs)]
    metrics = df['metric'].unique()
    models = df['name'].unique()
    fig, ax = plt.subplots(
        len(metrics),
        len(models),
        figsize=(len(models) * 5, len(metrics) * 3),
        squeeze=False,
        sharex='col',
        sharey='row',
    )
    for metric_id, metric_name in enumerate(metrics):
        for model_id, model_name in enumerate(models):
            df_plot = df.query(f'metric == "{metric_name}" and name == "{model_name}"')
            df_plot = df_plot.drop(columns=['metric', 'name'])
            axis = ax[metric_id][model_id]
            axis.margins(x=0.02)
            if agg_run:
                plot_agg_runs(axis, df_plot)
            else:
                plot_all_runs(axis, df_plot)

    for metric_id, metric_name in enumerate(metrics):
        ax[metric_id, 0].set_ylabel(
            metric_name,
            fontweight='bold',
            fontsize=16,
            rotation=90,
            va='bottom',
        )
    for model_id in range(len(models)):
        ax[-1, model_id].set_xlabel('epoch', fontsize=12)
    for row in range(0, len(metrics), 5):
        for model_id, model_name in enumerate(models):
            ax[row, model_id].set_title(model_name, fontweight='bold', fontsize=18, va='bottom')
    fig.suptitle(
        'Comparison of metrics per model and epoch',
        fontweight='bold',
        fontsize=24,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    return fig
