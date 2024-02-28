import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d

from uq import utils

from .dataframes import build_test_metric_accessor, set_hparams_columns

log = utils.get_logger(__name__)


def mscatter(x, y, z=None, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers

    if not ax:
        ax = plt.gca()
    if z is None:
        sc = ax.scatter(x, y, **kw)
    else:
        sc = ax.scatter3D(x, y, z, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def add_arrow(c_tuple, line=None, axis=None, size=5):
    color = line.get_color()
    c = np.stack(c_tuple, axis=1)
    annotations = []
    for i in range(len(c) - 1):
        p1, p2 = c[i], c[i + 1]
        mid = (p1 + p2) / 2
        d = p2 - p1
        mid2 = mid + d / 1000
        if mid.shape[0] == 2:
            midx, midy = mid
            mid2x, mid2y = mid2
        else:
            midx, midy, _ = proj3d.proj_transform(*mid, axis.get_proj())
            mid2x, mid2y, _ = proj3d.proj_transform(*mid2, axis.get_proj())

        annotation = line.axes.annotate(
            '',
            xytext=(midx, midy),
            xy=(mid2x, mid2y),
            arrowprops=dict(arrowstyle='->', color=color, lw=0.3),
            size=size,
        )
        annotations.append(annotation)
    return annotations


def add_outer_legend(fig, xlabel, ylabel, fontsize=11):
    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.grid(False)
    big_ax.set_xlabel(xlabel, fontsize=fontsize)
    big_ax.yaxis.labelpad = 10
    big_ax.set_ylabel(ylabel, fontsize=fontsize)


class HParamPlot:
    def __init__(
        self,
        df,
        xlabel='test_nll',
        ylabel=None,
        zlabel=None,
        hue=None,
        size=None,
        style=None,
        join=None,
        xlim=None,
        ylim=None,
        facets=None,
        ncols=3,
        fixed_hparams={},
    ):
        self.df = df.copy().reset_index()
        assert len(self.df) > 0, 'The dataframe must not be empty'
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.hue = hue
        self.size = size
        self.style = style
        self.join = join
        self.xlim = xlim
        self.ylim = ylim
        self.facets = facets
        self.ncols = ncols
        self.fixed_hparams = fixed_hparams

    def show(self):
        self.init_df()
        self.init_maps()
        self.filter_cols()
        scatter = self.plot()
        legend = self.add_legend(scatter)
        self.adjust_legend_titles(legend)
        self.fig.tight_layout()
        return self.fig

    def init_df(self):
        self.df[''] = 0
        if self.ylabel is None:
            self.ylabel = ''
        if self.facets is None:
            self.facets = ''
        self.nfacets = self.df[self.facets].nunique()
        self.axes = list({self.xlabel, self.ylabel, self.zlabel, self.facets} - {None})
        self.groups = list({self.hue, self.size, self.style, self.join} - {None})
        for col in self.axes + self.groups:
            assert col in self.df.columns, f'The column {col} is not present in the dataframe.'

        for key, value in self.fixed_hparams.items():
            if type(value) == list:
                self.df = self.df[self.df[key].isin(value)]
            else:
                self.df = self.df[self.df[key] == value]
        self.df = self.df.sort_values(self.groups)

    def init_maps(self):
        if self.hue:
            cm = mpl.cm.get_cmap('magma')
            color_keys = self.df[self.hue].unique().tolist()
            color_linear = np.linspace(1, 0, len(color_keys) + 2)[1:-1]
            color_values = [cm(l) for l in color_linear]
            self.listed_cm = mpl.colors.ListedColormap(color_values)
            color_ids = list(range(len(color_keys)))
            self.c_map = dict(zip(color_keys, color_ids))

        if self.size:
            s_keys = self.df[self.size].unique()
            # s_values = [x**2 for x in range(6, 16 + 2, 2)]
            # s_values = [int(x**1.2) for x in range(6, 30 + 2, 2)]
            s_values = [15, 20, 28, 40]
            assert len(s_keys) <= len(s_values)
            s_values = s_values[: len(s_keys)]
            self.s_map = dict(zip(s_keys, s_values))

        if self.style:
            m_keys = self.df[self.style].unique()
            m_values = [
                'o',
                'X',
                's',
                'P',
                'p',
                'D',
                'v',
                '^',
                '<',
                '>',
                '*',
                'd',
                'h',
                'H',
            ]
            assert len(m_keys) <= len(m_values)
            m_values = m_values[: len(m_keys)]
            self.m_map = dict(zip(m_keys, m_values))

    def filter_cols(self):
        cols = self.axes + self.groups
        self.df = self.df[list(set(cols))]

    def plot(self):
        # Plot the smaller points above larger points
        # self.df = self.df.sort_values(self.size, kind='stable', ascending=False)
        projection = '3d' if self.zlabel else None
        # self.axis = plt.axes(projection=projection)
        ncols = min(self.ncols, self.nfacets)
        nrows = math.ceil(self.nfacets / ncols)
        figsize = (ncols * 4, nrows * 3) if self.ylabel else (ncols * 4, nrows * 0.5)
        self.fig = plt.figure(figsize=figsize, dpi=300)
        for i, (facet, facet_df) in enumerate(self.df.groupby(self.facets)):
            axis = self.fig.add_subplot(nrows, ncols, i + 1, projection=projection)
            scatter = self.plot_axis(axis, facet_df, facet)
        return scatter

    def plot_axis(self, axis, df, facet):
        kwargs = {}
        if self.hue:
            kwargs['c'] = df[self.hue].map(self.c_map)
            kwargs['cmap'] = self.listed_cm
        if self.size:
            kwargs['s'] = df[self.size].map(self.s_map)
        if self.style:
            kwargs['m'] = df[self.style].map(self.m_map)
        if self.zlabel:
            kwargs['z'] = df[self.zlabel]
        scatter = mscatter(
            x=df[self.xlabel],
            y=df[self.ylabel],
            ax=axis,
            edgecolor='white',
            linewidth=0.3,
            **kwargs,
        )

        if self.join:
            self.to_remove = []
            self.update_join(axis, df)
            self.fig.canvas.mpl_connect('motion_notify_event', lambda e: self.update_join(axis, df))
        # axis.set(xlabel=self.xlabel, ylabel=self.ylabel)
        # if self.zlabel:
        #     axis.set(zlabel=self.zlabel)
        labels_fontsize = 9
        if self.df[self.facets].nunique() > 1:
            labels_fontsize = 13
        add_outer_legend(self.fig, self.xlabel, self.ylabel, fontsize=labels_fontsize)
        axis.tick_params(axis='both', which='major', labelsize=8)
        axis.tick_params(axis='both', which='minor', labelsize=6)
        if self.facets != '':
            axis.set_title(facet, fontsize=10)
        axis.set(xlim=self.xlim, ylim=self.ylim)
        if 'calib_' in self.ylabel:
            axis.set_ylim(bottom=0)
        if self.ylabel == '':
            axis.set_yticks([])
        return scatter

    def update_join(self, axis, df):
        self.remove_join()
        self.add_join(axis, df)

    def remove_join(self):
        for obj in self.to_remove:
            obj.remove()
        self.to_remove = []

    def add_join(self, axis, df):
        join_groups = list(set(self.groups) - {self.join})
        if join_groups:
            dfs_grouped = df.groupby(list(set(self.groups) - {self.join}))
        else:   # Handle the case where there are no groups
            dfs_grouped = [(None, df)]
        for g, df_grouped in dfs_grouped:
            df_grouped = df_grouped[df_grouped[self.join].notna()]
            assert df_grouped[self.join].is_unique
            df_grouped = df_grouped.sort_values(self.join)
            kwargs = {}
            if self.zlabel:
                kwargs['zs'] = df_grouped[self.zlabel]
            coords = (
                df_grouped[self.xlabel].to_numpy(),
                df_grouped[self.ylabel].to_numpy(),
            )
            if self.zlabel:
                coords += (df_grouped[self.zlabel].to_numpy(),)
            line = axis.plot(*coords, color='black', linewidth=0.2)[0]
            self.to_remove.append(line)
            annotations = add_arrow(coords, line, axis, size=5)
            self.to_remove.extend(annotations)

    def add_legend(self, scatter):
        handles = []
        labels = []

        def add_title(title):
            nonlocal handles
            nonlocal labels
            handles += [Line2D([], [], color='none')]
            new_title = title.replace('_', r'\_')
            labels += [rf'$\bf{{{new_title}}}$']

        if self.hue:
            num = len(self.c_map)
            new_handles = scatter.legend_elements(prop='colors', num=num)[0][:num]
            if len(new_handles) == num:
                add_title(self.hue)
                handles += new_handles
                labels += list(self.c_map)
            else:
                log.warn("scatter.legend_elements(prop='colors') gave incorrect handles")
        if self.size:
            num = len(self.s_map)
            new_handles = scatter.legend_elements(prop='sizes', num=num)[0][:num]
            if len(new_handles) == num:
                add_title(self.size)
                handles += new_handles
                labels += list(self.s_map)
            else:
                log.warn("scatter.legend_elements(prop='sizes') gave incorrect handles")
        if self.style:
            add_title(self.style)
            handles += [
                Line2D([], [], color='black', marker=m, linestyle='None') for m in self.m_map.values()
            ]
            labels += list(self.m_map)
        if self.join:
            add_title(self.join)
            j_labels = self.df[self.join].unique()
            handles += [Line2D([], [], color='none', label='') for _ in range(len(j_labels))]
            labels += list(j_labels)
        legend_fontsize = 10 if self.nfacets > 1 else 5.5
        legend = self.fig.legend(
            handles,
            labels,
            bbox_to_anchor=(1, 0.5),
            loc='center left',
            borderaxespad=0.0,
            fontsize=legend_fontsize,
        )
        legend.set_draggable(True)
        return legend

    def adjust_legend_titles(self, legend):
        # Move titles to the left
        for item, label in zip(legend.legendHandles, legend.texts):
            if r'\bf{' in label._text:
                width = item.get_window_extent(self.fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-3 * width, 0))


def plot_hparam_sns(df, fixed_hparams={}, **kwargs):
    df = df.copy()
    assert len(df) > 0, 'The dataframe must not be empty'
    set_hparams_columns(df)
    # Filter the rows to keep (https://stackoverflow.com/a/34162576)
    to_keep = (df[list(fixed_hparams)] == pd.Series(fixed_hparams, dtype=object)).all(axis=1)
    df = df.loc[to_keep]
    df = df[df['run_id'] == 0]
    xlabel = 'test_nll'
    ylabel = 'test_calib_l1'
    for metric in [xlabel, ylabel]:
        df[metric] = df['metrics'].map(build_test_metric_accessor(metric))

    # hp_color = 'K'
    # cm = mpl.cm.get_cmap('magma')
    # color_keys = df[hp_color].unique().tolist()
    # color_linear = np.linspace(1, 0, len(color_keys) + 2)[1:-1]
    # color_values = [cm(l) for l in color_linear]
    # listed_cm = mpl.colors.ListedColormap(color_values)

    fig, axis = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    sns.scatterplot(data=df, ax=axis, **kwargs)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)


def plot_3d(df):
    x = 'test_nll'
    y = 'test_stddev'
    z = 'test_calib_l1'

    for metric in [x, y, z]:
        df[metric] = df['metrics'].map(build_test_metric_accessor(metric))

    # fig, axis = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(df[x], df[y], df[z])
