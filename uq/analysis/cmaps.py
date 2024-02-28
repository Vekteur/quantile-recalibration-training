import seaborn as sns
import colorcet as cc

from .constants import base_model_names, cat_cmap


def posthoc_or_regul_cmap(df):
    blues = sns.color_palette('Blues', 4)[1:]
    greens = sns.color_palette('Greens', 4)[1:]
    reds = sns.color_palette('Reds', 4)[1:]

    method_map = {
        'KDE Recal': blues[0],
        'Linear Recal': blues[1],
        'CQR': blues[1],
        'Recal': blues[2],
        'QR': greens[0],
        'Trunc': greens[0],
        'PCE-KDE': greens[1],
        'PCE-Sort': greens[2],
    }
    cmap = {}
    for base_loss in df['base_loss'].unique():
        for name, color in method_map.items():
            cmap[f'{base_model_names[base_loss]} + {name}'] = color
        cmap[f'{base_model_names[base_loss]}'] = reds[1]
    return cmap


def inhoc_or_posthoc_or_regul_cmap(df):
    blues = sns.color_palette('Blues', 3)[1:]
    greens = sns.color_palette('Greens', 4)[1:]
    reds = sns.color_palette('Reds', 2)[1:]

    method_map = {
        'RecTr (ca)': greens[0],
        'RecTr (tr)': greens[1],
        'RecTr (tr) + Rec (ca)': greens[2],
        'Rec (tr)': blues[0],
        'Rec (ca)': blues[1],
        'PCE-KDE': reds[0],
    }
    cmap = {}
    for name, color in method_map.items():
        cmap[name] = color
    return cmap


def posthoc_dataset_cmap(df):
    blues = sns.color_palette('Blues', 4)[1:]
    greens = sns.color_palette('Greens', 4)[1:]
    reds = sns.color_palette('Reds', 4)[1:]

    names = df.name.unique()
    end = ' (calib)'
    posthoc_names = [name[: -len(end)] for name in names if name.endswith(end)]
    cmap = {}
    for i, posthoc_name in enumerate(posthoc_names):
        cmap[f'{posthoc_name} (train)'] = blues[i]
        cmap[f'{posthoc_name} (calib)'] = greens[i]
    i = 0
    for name in names:
        if name not in cmap:
            cmap[name] = reds[i]
            i += 1
    return cmap


def inhoc_alpha_cmap(df):
    inhoc_alphas = df['inhoc_alpha'].dropna().sort_values().unique()
    gradient = sns.color_palette('flare', len(inhoc_alphas))
    cmap = {
        rf'$\alpha$ = {inhoc_alpha}': gradient[i] for i, inhoc_alpha in enumerate(inhoc_alphas)
    }
    return cmap


def inhoc_b_cmap(df):
    inhoc_bs = df['inhoc_b'].dropna().sort_values().unique()
    gradient = sns.color_palette('flare', len(inhoc_bs))
    cmap = {rf'$b = {inhoc_b}$': gradient[i] for i, inhoc_b in enumerate(inhoc_bs)}
    cmap['Tuned $b$'] = cat_cmap[0]
    return cmap


def inhoc_cal_size(df):
    inhoc_cal_sizes = df['inhoc_cal_size'].dropna().sort_values().unique()
    gradient = sns.color_palette('flare', len(inhoc_cal_sizes))
    cmap = {rf'QRTC-{inhoc_cal_size}': gradient[i] for i, inhoc_cal_size in enumerate(inhoc_cal_sizes)}
    cmap[r'\textbf{QRTC}'] = cat_cmap[0]
    return cmap


def lambda_cmap(df):
    lambdas = df['lambda_'].unique()
    gradient = sns.color_palette('flare', len(lambdas))
    cmap = {rf'$\lambda$ = {lambda_}': gradient[i] for i, lambda_ in enumerate(lambdas)}
    cmap[r'Tuned $\lambda$'] = sns.color_palette('Reds', 4)[2]
    return cmap 


def base_loss_cmap(df):
    blues = sns.color_palette('Blues', 4)[1:]
    greens = sns.color_palette('Greens', 4)[1:]
    reds = sns.color_palette('Reds', 4)[1:]

    names = df.name.unique()
    end = ' (calib)'
    posthoc_names = [name[: -len(end)] for name in names if name.endswith(end)]
    cmap = {}
    for i, posthoc_name in enumerate(posthoc_names):
        cmap[f'{posthoc_name} (train)'] = blues[i]
        cmap[f'{posthoc_name} (calib)'] = greens[i]
    i = 0
    for name in names:
        if name not in cmap:
            cmap[name] = reds[i]
            i += 1

    return cmap


def get_cmap(df, cmap):
    names = df.name.unique()
    if cmap is None:
        return {name: color for name, color in zip(names, sns.color_palette(n_colors=len(names)))}
    elif type(cmap) == str:
        cmap = {
            'posthoc_or_regul': posthoc_or_regul_cmap,
            'posthoc_dataset': posthoc_dataset_cmap,
            'inhoc_or_posthoc_or_regul': inhoc_or_posthoc_or_regul_cmap,
            'inhoc_alpha': inhoc_alpha_cmap,
            'inhoc_b': inhoc_b_cmap,
            'inhoc_cal_size': inhoc_cal_size,
            'lambda_': lambda_cmap,
            'base_loss': base_loss_cmap,
        }[cmap](df)
    return cmap
