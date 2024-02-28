import pandas as pd
import numpy as np
import seaborn as sns


ext = 'pdf'
main_metrics = ['test_nll', 'test_calib_l1', 'test_wis', 'test_stddev']
#other_metrics_metrics = ['test_calib_l2', 'test_rmse', 'test_mae']
other_metrics = ['test_quantile_score_0.05', 'test_quantile_score_0.25', 'test_quantile_score_0.50', 'test_quantile_score_0.75', 'test_quantile_score_0.95']

cat_cmap = sns.color_palette()

misspec_queries = {
    'small_mlp': 'misspecification == "small_mlp" and misspecification.notna()',
    'big_mlp': 'misspecification == "big_mlp" and misspecification.notna()',
    'homoscedasticity': 'misspecification == "homoscedasticity" and misspecification.notna()',
    'sharpness_reward': 'misspecification == "sharpness_reward" and misspecification.notna()',
    'mixture_size_1': 'mixture_size == 1 and mixture_size.notna()',
    'drop_prob_0_9': 'drop_prob == 0.9 and drop_prob.notna()',
}
misspec_queries = {key: f'({value})' for key, value in misspec_queries.items()}

any_misspec_query = ' or '.join(misspec_queries.values())
any_misspec_query = f'({any_misspec_query})'
no_misspec_query = f'(not {any_misspec_query})'

all_misspec_queries = {f'misspec_{name}': query for name, query in misspec_queries.items()}
all_misspec_queries['no_misspec'] = no_misspec_query

regul_names = {
    'no_regul': None,
    'entropy_based': 'QR',
    'entropy_based_mc': 'QR_mc',
    'entropy_based_ss': 'QR_ss',
    'RT_mc': 'RT_mc',
    'RT_ss': 'RT_ss',
    'cdf_based': 'PCE-KDE',
    'quantile_based': 'PCE-Sort',
    'truncated': 'Trunc',
    'oqr': 'OQR',
    'ic': 'IC',
}
posthoc_names = {
    'ecdf': 'RecStep',
    'smooth_ecdf': 'Rec',  #'KDE Recal'
    'stochastic_ecdf': 'Stochastic Recal',
    'linear_ecdf': 'Linear Recal',
    'CQR': 'CQR',
    'conditional': 'Auto',
}

inhoc_variant_names = {
    None: 'Tr',
    pd.NA: 'Tr',
    np.nan: 'Tr',
    'only_init': 'In',
    'no_grad': 'NG',
    'learned': 'Ln',
}
metric_names = {
    'nll': 'NLL',
    'crps': 'CRPS',
    'wis': 'CRPS',
    'calib_l1': 'PCE',
    'calib_l2': 'PCE_2',
    'stddev': 'SD',
    'mae': 'MAE',
    'rmse': 'RMSE',
    'quantile_score_0.05': 'QS at level 0.05',
    'quantile_score_0.25': 'QS at level 0.25',
    'quantile_score_0.50': 'QS at level 0.50',
    'quantile_score_0.75': 'QS at level 0.75',
    'quantile_score_0.95': 'QS at level 0.95',
}
base_model_names = {
    'nll': 'MIX-NLL',
    'crps': 'MIX-CRPS',
    'expected_qs': 'SQR-CRPS',
}
dataset_names = {
    'train': 'tr',
    'calib': 'ca',
    'batch': 'ba',
}
