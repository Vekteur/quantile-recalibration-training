import pandas as pd


def make_test_df_mean(test_df):
    test_df = test_df.reset_index(level='run_id', drop=True)
    return test_df.groupby(test_df.index.names, dropna=False).mean()


def select_best_hparam(df, hparam, differ_by, metric_to_minimize='val_nll'):
    """
    df is a dataframe with the dataset, run_id and hyperparameters on the index.
    Groups of runs to be tuned are formed by grouping runs with same index, except the tuned hyperparameter and 
    the hyperparameters in differ_by.
    """

    # For more stable results, we compare the *mean* NLL per model
    df_mean = make_test_df_mean(df[[metric_to_minimize]])
    groupby = set(df_mean.index.names) - differ_by
    # We select the row with the best posthoc_b for each model
    df_mean = df_mean.reset_index()   # Resetting the index is necessary
    idx_min = df_mean.groupby(list(groupby - {hparam}), dropna=False)[metric_to_minimize].idxmin(skipna=False)
    idx_min = idx_min.dropna()
    selected_rows = df_mean.loc[idx_min]
    # We select the corresponding rows in the original dataframe
    index_names = df.index.names
    selected_rows = selected_rows.drop(columns=metric_to_minimize)
    df = selected_rows.merge(df.reset_index(), on=list(groupby))
    df = df.set_index(index_names)
    return df, selected_rows


def duplicate_baseline_per_regul(df, hparam, baseline_query, default_hparam_value, differ_by, baseline_differ_by):
    """
    df is a dataframe with the dataset, run_id and hyperparameters on the index.
    Groups of runs to be tuned are formed by grouping runs with same index, except the tuned hyperparameter and 
    the hyperparameters in differ_by.
    For each group, the corresponding baseline, which must only differ by the hyperparameters in baseline_differ_by,
    is duplicated in this group with the corresponding hyperparameteres of the group and the default value for the 
    tuned hyperparameter.
    """

    baseline_mask = df.eval(baseline_query)
    df_baseline = df[baseline_mask]
    df_regul = df[~baseline_mask]
    groupby = set(df.index.names) - {hparam} - differ_by
    # Get all regularization groups
    groups = df_regul.groupby(list(groupby), dropna=False).size()
    # Convert the result to a dataframe
    groups = groups.index.to_frame().reset_index(drop=True)
    # Get the baseline for each regularization group
    #display('groups:', groups)
    #display('df_baseline:', df_baseline)
    match_list = list(groupby - baseline_differ_by)
    if len(groups) > 0:
        baseline_matches = len(groups.merge(df_baseline, how='inner', on=match_list)) > 0
        assert baseline_matches, f'No baseline matches a group of regularization methods. Is this match_list correct? {match_list}'
    df_baseline_per_regul = groups.merge(df_baseline, how='left', on=match_list)
    #display('df_baseline_per_regul:', df_baseline_per_regul)
    df_baseline_per_regul[hparam] = default_hparam_value
    index = df_regul.index.names
    concat = pd.concat([df_regul.reset_index(), df_baseline_per_regul])
    return concat.set_index(index)


def select_best_hparam_with_max_wis_loss(df, hparam, baseline_query, differ_by, accepted_relative_wis_loss):
    """
    df is a dataframe with the dataset, run_id and hyperparameters on the index.
    Groups of runs to be tuned are formed by grouping runs with same index, except the tuned hyperparameter and 
    the hyperparameters in differ_by.
    """

    # For more stable results, we compare the *mean* WIS and calib_l1 per model
    df_mean = make_test_df_mean(df[['val_wis', 'val_calib_l1']])
    groupby = set(df_mean.index.names) - differ_by
    # We apply the constraint on WIS (models without regularization are guaranteed to be selected).
    # Hyperparameters are selected if their WIS is not more than accepted_relative_wis_loss% worse than the corresponding baseline.
    baseline = df_mean.query(baseline_query)[['val_wis']]
    df_mean = df_mean.reset_index(level=[hparam])
    # display(baseline)
    # display(df_mean)
    df_mean = df_mean.merge(
        baseline,
        how='left',
        on=list(groupby - {hparam}),
        suffixes=(None, '_baseline'),
        validate='many_to_one',
    )
    # display(df_mean)
    mask = df_mean['val_wis'] <= df_mean['val_wis_baseline'] * (1 + accepted_relative_wis_loss)
    df_mean = df_mean[mask]
    # Take hparam with minimum calibration per model
    df_mean = df_mean.reset_index()
    idxmin = df_mean.groupby(list(groupby - {hparam}), dropna=False)['val_calib_l1'].idxmin(skipna=False)
    selected_rows = df_mean.loc[idxmin]
    # We select the corresponding rows in the original dataframe
    index_names = df.index.names
    df = selected_rows.merge(df.reset_index(), on=list(groupby))
    df = df.set_index(index_names)
    return df, selected_rows
