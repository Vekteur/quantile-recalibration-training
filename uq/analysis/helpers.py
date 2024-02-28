from functools import partial

import numpy as np
import pandas as pd
from IPython.display import display

from uq.analysis.dataframes import get_duplication_df, make_df_abb
from uq.utils.general import filter_dict, set_notebook_options, savefig, op_without_index
from uq.analysis.df_tuning import select_best_hparam, duplicate_baseline_per_regul, select_best_hparam_with_max_wis_loss
from uq.analysis.plot_cohen_d import build_cohen_d, plot_cohen_d_boxplot, build_diff_df, plot_metric_difference
from uq.analysis.plot_cd_diagram import draw_my_cd_diagram
from uq.analysis.constants import ext, main_metrics, other_metrics


def standard_setting(config, df, remove_dup_index=True, without_discrete=False, add_abb=False, **kwargs):
    # Remove duplicate runs if any
    if remove_dup_index:
        df = df[~df.index.duplicated(keep='first')]
    # Optionally remove the most discrete datasets
    if without_discrete:
        df_dup = get_duplication_df(config)
        discrete_datasets = df_dup.query('`Proportion of top 10 classes` > 0.5').reset_index().dataset
        df = df.query('dataset not in @discrete_datasets')
    if add_abb:
        df_abb = make_df_abb(df['dataset'].unique().astype(str))
        df_abb = op_without_index(df, lambda df: df.merge(df_abb, on='dataset'))
    df = add_name_index(df, **kwargs)
    df = op_without_index(df, latex_names)
    return df


def preprocess(df):
    def op(df):
        if 'inhoc_b' in df.columns:
            mask = df.eval('inhoc_method == "smooth_ecdf" and inhoc_b.isna()')
            df[mask] = df[mask].eval('inhoc_b = 0.1')
        if 'lambda_' in df.columns:
            mask = df.eval('regul.str.startswith("entropy_based") and lambda_ == -1')
            df.loc[mask, 'regul'] = df.loc[mask, 'regul'].str.replace('entropy_based', 'RT')
            df.loc[mask, 'lambda_'] = np.nan
            mask = df.eval('regul.str.startswith("entropy_based") and lambda_ == 0')
            df.loc[mask, 'regul'] = np.nan
            df.loc[mask, 'lambda_'] = np.nan
        if 'inhoc_alpha' in df.columns:
            mask = df.eval('inhoc_method == "smooth_ecdf" and inhoc_alpha.isna()')
            df.loc[mask, 'inhoc_alpha'] = 1
            mask = df.eval('inhoc_method == "smooth_ecdf" and inhoc_alpha > 0 and inhoc_alpha != 1')
            df.loc[mask, 'inhoc_method'] = 'OTHER'
            mask = df.eval('inhoc_method == "smooth_ecdf" and inhoc_alpha == 0')
            # We remove these runs because they are duplicates
            df.loc[mask] = np.nan
            mask = df.eval('inhoc_method == "smooth_ecdf" and inhoc_alpha < 0')
            df.loc[mask, 'inhoc_method'] = 'QREG'
            mask = df.eval('inhoc_method == "smooth_ecdf" and inhoc_alpha == 1')
            df.loc[mask, 'inhoc_method'] = 'QRT'
            mask = df.eval('inhoc_method.isna()')
            df.loc[mask, 'inhoc_alpha'] = 0
        if 'inhoc_dataset' not in df.columns:
            df['inhoc_dataset'] = np.nan
        if 'inhoc_reflected' in df.columns:
            mask = df.eval('not inhoc_method.isna() and inhoc_reflected.isna()')
            df.loc[mask, 'inhoc_reflected'] = True
            mask = df.eval('not inhoc_method.isna() and inhoc_truncated.isna()')
            df.loc[mask, 'inhoc_truncated'] = False
        if 'posthoc_reflected' in df.columns:
            mask = df.eval('not posthoc_method.isna() and posthoc_reflected.isna()')
            df.loc[mask, 'posthoc_reflected'] = True
            mask = df.eval('not posthoc_method.isna() and posthoc_truncated.isna()')
            df.loc[mask, 'posthoc_truncated'] = False
        if 'cal_size' in df.columns:
            mask = df.eval('cal_size.isna()')
            df.loc[mask, 'cal_size'] = 2048
        mask = df.eval('inhoc_dataset.isna()')
        df.loc[mask, 'inhoc_dataset'] = 'batch'
        if 'test_nll' in df.columns: # Remove runs that diverge
            df = df.query('test_nll < 1000')
        return df

    df = op_without_index(df, op)
    if 'regul' not in df.index.names and 'regul' not in df.columns:
        df['regul'] = np.nan
        df = df.set_index('regul', append=True)
    return df


def filter_df(plot_df, mixture_size=[3], cal_size=[2048], inhoc_cal_size=[2048],
        only_inhoc_reflected=True, only_posthoc_reflected=True, 
        keep_non_tuned_inhoc_b=False, no_inhoc_variant=True, 
        inhoc_dataset=['batch'], base_model='nn', all_base_loss=False,
        inhoc_alpha_to_tune=True,
    ):

    def has_col(col):
        return col in plot_df.index.names or col in plot_df.columns

    plot_df = plot_df.query('posthoc_method != "ecdf"')
    if mixture_size is not None:
        plot_df = plot_df.query('mixture_size in @mixture_size')
    if cal_size is not None and has_col('cal_size'):
        plot_df = plot_df.query('cal_size.isna() or cal_size in @cal_size')
    if only_inhoc_reflected and has_col('inhoc_reflected'):
        plot_df = plot_df.query('inhoc_reflected.isna() or inhoc_reflected')
        plot_df = plot_df.query('inhoc_truncated.isna() or not inhoc_truncated')
    if only_posthoc_reflected and has_col('posthoc_reflected'):
        plot_df = plot_df.query('posthoc_reflected.isna() or posthoc_reflected')
        plot_df = plot_df.query('posthoc_truncated.isna() or not posthoc_truncated')
    if no_inhoc_variant and has_col('inhoc_variant'):
        plot_df = plot_df.query('inhoc_variant.isna()')
    if inhoc_dataset is not None:
        plot_df = plot_df.query('inhoc_dataset.isna() or inhoc_dataset in @inhoc_dataset')
    if base_model is not None:
        plot_df = plot_df.query('base_model.isna() or base_model == @base_model')
    if inhoc_cal_size is not None and has_col('inhoc_cal_size'):
        plot_df = plot_df.query('inhoc_cal_size.isna() or inhoc_cal_size in @inhoc_cal_size')
    
    if not all_base_loss:
        plot_df = plot_df.query('base_loss != "nll_inhoc_ss"')
    if inhoc_alpha_to_tune and has_col('inhoc_alpha'):
        plot_df = tune_inhoc_alpha(plot_df)
    if has_col('inhoc_b'):
        plot_df = tune(plot_df, 'inhoc_b', keep_non_tuned=keep_non_tuned_inhoc_b)
    return plot_df


def plot_cohen_d_and_cd_diagrams(config, test_df, baseline_query, join_by, path, main_metrics=main_metrics, 
        other_metrics=None, cmap=None, cohen_kwargs={}, diff_kwargs={}, cd_kwargs={}, **kwargs):
    
    if cmap is not None:
        if cohen_kwargs is not None:
            cohen_kwargs['cmap'] = cmap
        if diff_kwargs is not None:
            diff_kwargs['cmap'] = cmap
    # Main metrics
    plot_df = standard_setting(config, test_df, **kwargs)
    df_cohen = build_cohen_d(plot_df, main_metrics, baseline_query, join_by)
    plot_cohen_d_boxplot(df_cohen, main_metrics, **cohen_kwargs)
    savefig(path / 'cohen_d' / f'main_metrics.{ext}')
    # CD diagrams
    if cd_kwargs is not None:
        for metric in main_metrics:
            draw_my_cd_diagram(plot_df, metric, **cd_kwargs)
            savefig(path / 'cd_diagrams' / f'{metric}.{ext}', dpi=300)
    if diff_kwargs is not None:
        for metric in main_metrics:
            df_diff = build_diff_df(plot_df, [metric], baseline_query, join_by)
            plot_metric_difference(df_diff, metric, **diff_kwargs)
            savefig(path / 'diff' / f'{metric}.{ext}')
    # Other metrics
    if other_metrics is not None:
        df_cohen = build_cohen_d(plot_df, other_metrics, baseline_query, join_by)
        plot_cohen_d_boxplot(df_cohen, other_metrics, **cohen_kwargs)
        savefig(path / 'cohen_d' / f'other_metrics.{ext}')


def plot_all_cohen_d(config, name, test_df, baseline_query, join_by, path, **kwargs):
    print(f'Plotting {name}')
    for without_discrete in [False, True]:
        sub_path = path / name
        if without_discrete:
            sub_path /= 'without_discrete'
        plot_cohen_d_and_cd_diagrams(config, test_df, baseline_query, join_by, path=sub_path, without_discrete=without_discrete, **kwargs)


def tune(test_df, hparam, keep_non_tuned=False, values=[0.02, 0.05, 0.1, 0.2, 0.5, 1], print_selected=False):
    if hparam not in test_df.reset_index().columns:
        print(f'Warning: column {hparam} not in dataframe')
        return test_df
    test_df = test_df.query(f'{hparam}.isna() or {hparam} in @values')
    to_tune = test_df.eval(f'not {hparam}.isna()')
    differ_by = {'lambda_', 'b', 's'}
    test_df_tuned, _ = select_best_hparam(test_df[to_tune], hparam=hparam, differ_by=differ_by, metric_to_minimize='val_nll')
    if keep_non_tuned:
        # We distinguish between tuned and non-tuned models by setting the hparam of the tuned model to NA
        def op(df):
            df[hparam] = pd.NA
            return df
        test_df_tuned = op_without_index(test_df_tuned, op)
    else:
        test_df = test_df[~to_tune]
    if print_selected:
        display(test_df_tuned.reset_index()[hparam].value_counts(dropna=False))
    return pd.concat([test_df_tuned, test_df])


def tune_inhoc_alpha(test_df, keep_non_tuned=False, print_selected=False):
    hparam = 'inhoc_alpha'
    differ_by = {'inhoc_b'}
    # Query that distinguishes between baseline and regularized models
    baseline_query = f'{hparam} == 0'
    # Query that isolates the models that do not need tuning
    # It must be exclusive with `baseline_query`
    isolate_query = 'inhoc_alpha > 0'
    isolate_mask = test_df.eval(isolate_query)
    # Columns that differ between the baseline and the regularized model, in addition to `hparam`
    #differ_by = {'regul', 's', 'L', 'spacing', 'neural_sort', 'divergence'}
    baseline_differ_by = {
        'inhoc_method', 'inhoc_dataset', 'inhoc_b', 'inhoc_reflected', 'inhoc_truncated',
        'L', 'spacing', 'neural_sort', 's', 'divergence',
        'base_loss', 'cal_size', 'inhoc_cal_size', 'posthoc_cal_size',
        'posthoc_dataset',
    }
    test_df_dup = duplicate_baseline_per_regul(
        test_df[~isolate_mask], 
        hparam, 
        baseline_query, 
        differ_by=differ_by,
        default_hparam_value=0, 
        baseline_differ_by=baseline_differ_by,
    )
    test_df_tuned, _ = select_best_hparam_with_max_wis_loss(
        test_df_dup, 
        hparam,
        baseline_query,
        differ_by=differ_by,
        accepted_relative_wis_loss=0.1
    )

    if keep_non_tuned:
        # We distinguish between tuned and non-tuned models by setting the hparam of the tuned model to NA
        def op(df):
            df[hparam] = pd.NA
            return df
        test_df_tuned = op_without_index(test_df_tuned, op)
    else:
        test_df = pd.concat([test_df.query(baseline_query), test_df[isolate_mask]])
    if print_selected:
        display(test_df_tuned.reset_index()[hparam].value_counts(dropna=False))
    return pd.concat([test_df_tuned, test_df])



def add_name_index(df, **kwargs):
    df = df.copy()
    index = df.index.names
    df = df.reset_index()
    model_name_partial = partial(create_name_from_dict, **kwargs)
    df['name'] = df.apply(model_name_partial, axis='columns').astype('string')
    df = df.set_index(index + ['name'])
    return df


# def create_name_from_dict_old(d, add_inhoc_dataset=True, add_posthoc_dataset=True, add_inhoc_alpha=False, add_inhoc_b=False, add_lambda_=False):
#     for s in d.index:
#         assert type(s) == str, (s, type(s))
    
#     components = []

#     # Regularization
#     if 'regul' in d and not pd.isna(d['regul']):
#         regul = d['regul']
#         regul = regul_names[regul]
#         if add_lambda_:
#             if pd.isna(d['lambda_']):
#                 regul = 'Tuned $\lambda$'
#             else:
#                 regul = rf'$\lambda$ = {d["lambda_"]}'
#         if 'divergence' in d and not pd.isna(d['divergence']):
#             div = d['divergence']
#             regul = f'{regul}_{div}'
#         if 'mc_dataset' in d and not pd.isna(d['mc_dataset']):
#             mc = d['mc_dataset']
#             if mc != 'batch':
#                 mc = dataset_names[mc]
#                 regul += f' ({mc})'
#         if 'cal_size' in d and not pd.isna(d['cal_size']):
#             cal_size = d['cal_size']
#             regul += f' ({cal_size})'
#         components.append(regul)
#     # In-hoc
#     if 'inhoc_method' in d and not pd.isna(d['inhoc_method']):
#         method = d['inhoc_method']
#         #method = posthoc_names[method]

#         # variant = d['inhoc_variant']
#         # if pd.isna(variant):
#         #     variant = 'Tr'
#         # else:
#         #     variant = inhoc_variant_names[variant]
#         # method += variant

#         dataset = d['inhoc_dataset']
#         dataset = dataset_names[dataset]
#         datasets_list = []
#         if add_inhoc_dataset:
#             datasets_list.append(dataset)

#         if 'inhoc_mc_dataset' in d and not pd.isna(d['inhoc_mc_dataset']):
#             mc = d['inhoc_mc_dataset']
#             mc = dataset_names[mc]
#             datasets_list.append(mc)
#         method += f' ({", ".join(datasets_list)})' if datasets_list else ''
        
#         if add_inhoc_b:
#             if pd.isna(d['inhoc_b']):
#                 return 'Tuned b'
#             else:
#                 return f'b = {d["inhoc_b"]}'
#         # if add_inhoc_alpha:
#         #     inhoc_alpha = d['inhoc_alpha']
#         #     method += f' ({inhoc_alpha}, {dataset})'
#         #     return rf'$\alpha$ = {inhoc_alpha}'
#         components.append(method)
#     # Post-hoc
#     if 'posthoc_method' in d and not pd.isna(d['posthoc_method']):
#         method = d['posthoc_method']
#         method = posthoc_names[method]
#         # if d['posthoc_method'] == 'smooth_ecdf':
#         #     b = d['posthoc_b']
#         #     method += f' (b={b})'
#         # if add_posthoc_dataset:
#         #     dataset = d['posthoc_dataset']
#         #     method += f' ({dataset})'
#         if add_posthoc_dataset:
#             dataset = d['posthoc_dataset']
#             dataset = dataset_names[dataset]
#             method += f' ({dataset})'
#         components.append(method)
    
#     name = ' + '.join(components)
#     if name == '':
#         name = 'Base'
#     return name


def create_name_from_dict(d, add_inhoc_dataset=True, add_posthoc_dataset=True, only_inhoc_cal_size=False, 
                          only_posthoc_reflected=False, only_inhoc_alpha=False, only_inhoc_b=False,
                          add_estimation=False, add_base_model=False, add_mixture_size=False):
    if only_inhoc_cal_size:
        inhoc_cal_size = d['inhoc_cal_size']

        if pd.isna(d['inhoc_method']):
            return 'BASE'
        elif d['inhoc_dataset'] == 'batch':
            return 'QRTC'
        else:
            #return rf"$|D_\text{{QRT}}| = {cal_size}$"
            return f'QRTC-{inhoc_cal_size}'
    
    if only_inhoc_alpha and not pd.isna(d['posthoc_method']):
        inhoc_alpha = d['inhoc_alpha']
        if pd.isna(inhoc_alpha):
            inhoc_alpha = 0.
        return rf'$\alpha$ = {inhoc_alpha}'
    
    if only_inhoc_b and not pd.isna(d['inhoc_method']):
        inhoc_b = d['inhoc_b']
        if pd.isna(inhoc_b):
            return 'Tuned $b$'
        return f'$b = {inhoc_b}$'

    if 'inhoc_method' in d and not pd.isna(d['inhoc_method']):
        assert d['inhoc_dataset'] == 'batch', d['inhoc_dataset']
        method = d['inhoc_method']
        if 'inhoc_variant' in d and not pd.isna(d['inhoc_variant']):
            variant = d['inhoc_variant']
            assert method == 'QRT'
            if variant == 'only_init':
                method = 'QRI'
            elif variant == 'no_grad':
                method = 'QRG'
            elif variant == 'learned':
                method = 'QRL'
    
        if 'posthoc_method' in d and not pd.isna(d['posthoc_method']):
            assert d['posthoc_dataset'] == 'calib'
            method += 'C'
    else:
        if 'posthoc_method' in d and not pd.isna(d['posthoc_method']):
            method = 'QRC'
        else:
            method = 'BASE'

    if only_posthoc_reflected:
        if 'posthoc_reflected' not in d or pd.isna(d['posthoc_reflected']):
            return 'BASE'
        reflected = d['posthoc_reflected']
        truncated = d['posthoc_truncated']
        if reflected:
            method += r'-REFL'
        elif truncated:
            method += r'-TRUNC'
        else:
            method += r'-KDE'
        # if reflected:
        #     method += r'-$\Phi_\theta^\textrm{REFL}$'
        # elif truncated:
        #     method += r'-$\Phi_\theta^\textrm{TRUNC}$'
        # else:
        #     method += r'-$\Phi_\theta^\textrm{KDE}$'

    if add_estimation:
        if d['base_loss'] == 'nll_inhoc_mc':
            method += '-KDE'
        elif d['base_loss'] == 'nll_inhoc_ss':
            method += '-SS'
    
    if add_base_model:
        if d['base_model'] == 'nn':
            method += '-MLP'
        elif d['base_model'] == 'resnet':
            method += '-RESNET'
    
    if add_mixture_size:
        mixture_size = d['mixture_size']
        method += f'-{mixture_size}'
    
    # if method == 'QRTC':
    #     method = rf'\textbf{{{method}}}'

    return method


def latex_names(df):
    df = df.copy()
    #df['name'] = df['name'].str.replace('QRTC', r'\textbf{QRTC}')
    df['name'] = df['name'].str.replace(r'^QRTC$', r'\\textbf{QRTC}', regex=True)
    return df
