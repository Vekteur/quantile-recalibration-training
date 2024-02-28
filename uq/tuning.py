import collections

from uq.utils.hparams import HP, Join, Union


def get_tuning_for_QRT(config):
    mlp = Join(HP(base_model='nn'), HP(nb_hidden=[3]), HP(batch_size=512))
    resnet = Join(HP(base_model='resnet'), HP(batch_size=512))

    def default_tuning(
        base,
        *args,
        mixture_size=3,
        base_loss='nll',
        inhoc_grid=HP(method=None),
        posthoc_dataset='calib',
        posthoc_grid=HP(method=None)
    ):
        return Join(
            base,
            HP(pred_type='mixture'),
            *args,
            HP(mixture_size=mixture_size),
            HP(base_loss=base_loss),
            HP(inhoc_grid=inhoc_grid),
            HP(posthoc_dataset=posthoc_dataset),
            HP(posthoc_grid=posthoc_grid),
        )

    bs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    inhoc_grid_list = [
        Join(HP(method='smooth_ecdf'), HP(alpha=alpha), HP(b=b))
        for alpha in [1]
        for b in set(bs)
    ]

    inhoc_grid_list_alpha = [
        Join(HP(method='smooth_ecdf'), HP(alpha=alpha), HP(b=b))
        for alpha in [-5, -1, -0.5, -0.1, -0.05, -0.01, 0, 0.1, 0.5, 1, 2, 5]
        for b in set(bs)
    ]

    inhoc_grid_list_cal_size = [
        Join(HP(method='smooth_ecdf'), HP(alpha=alpha), HP(b=b), HP(cal_size=cal_size))
        for alpha in [1]
        for cal_size in [8, 32, 128, 512, 2048]
        for b in [0.01, 0.05, 0.1, 0.2]
    ]

    posthoc_grid = HP(method=[None, 'smooth_ecdf'])

    inhoc_grid_smoothing = [
        Join(HP(method='smooth_ecdf'), HP(alpha=1), HP(b=b), HP(reflected=False), HP(truncated=truncated))
        for truncated in [False, True]
        for b in set(bs)
    ]

    posthoc_grid_smoothing = Join(HP(method='smooth_ecdf'), HP(reflected=False), HP(truncated=[False, True]))

    return Union(
        ## Base
        # Note: when posthoc_dataset='train', calibration data is used for training the base model
        default_tuning(mlp, posthoc_dataset=['calib', 'train'], base_loss='nll', posthoc_grid=posthoc_grid),
        ## QRT
        default_tuning(mlp, HP(inhoc_dataset=['batch']), base_loss='nll_inhoc_mc', inhoc_grid=inhoc_grid_list, posthoc_grid=posthoc_grid),
        ## QRT with different alphas
        default_tuning(mlp, HP(inhoc_dataset=['batch']), base_loss='nll_inhoc_mc', inhoc_grid=inhoc_grid_list_alpha, posthoc_grid=posthoc_grid),
        ## QRT with sampling in training dataset and different calibration sizes
        default_tuning(mlp, HP(inhoc_dataset=['train']), base_loss='nll_inhoc_mc', inhoc_grid=inhoc_grid_list_cal_size, posthoc_grid=posthoc_grid),
        ## Base with other mixture sizes
        default_tuning(mlp, mixture_size=[1, 10], posthoc_dataset=['calib', 'train'], base_loss='nll', posthoc_grid=posthoc_grid),
        ## QRT with other mixture sizes
        default_tuning(mlp, HP(inhoc_dataset=['batch', 'calib']), mixture_size=[1, 10], base_loss='nll_inhoc_mc', inhoc_grid=inhoc_grid_list, posthoc_grid=posthoc_grid),
        ## Standard vs reflected vs truncated
        default_tuning(mlp, HP(inhoc_dataset=['batch']), base_loss='nll_inhoc_mc', inhoc_grid=inhoc_grid_smoothing, posthoc_grid=posthoc_grid),
        default_tuning(mlp, HP(cal_size=[2048]), HP(inhoc_dataset=['batch']), base_loss='nll_inhoc_mc', inhoc_grid=HP(method='smooth_ecdf'), posthoc_grid=posthoc_grid_smoothing),
        default_tuning(mlp, HP(cal_size=[2048]), base_loss='nll', posthoc_dataset='calib', posthoc_grid=posthoc_grid_smoothing),
        ## Ablation study
        default_tuning(
            mlp,
            HP(inhoc_variant=['only_init', 'no_grad', 'learned']),
            HP(inhoc_dataset='batch'),
            base_loss='nll_inhoc_mc',
            inhoc_grid=inhoc_grid_list,
            posthoc_grid=posthoc_grid,
        ),
        ## Base with ResNet
        default_tuning(resnet, posthoc_dataset=['calib', 'train'], base_loss='nll', posthoc_grid=posthoc_grid),
        ## QRT with ResNet
        default_tuning(resnet, HP(cal_size=[2048]), HP(inhoc_dataset=['batch']), base_loss='nll_inhoc_mc', inhoc_grid=inhoc_grid_list, posthoc_grid=posthoc_grid),
    )


def get_tuning_for_QRT_per_epoch(config):
    mlp = Join(HP(base_model='nn'), HP(nb_hidden=[3]), HP(batch_size=512))
    dist = Join(
        mlp,
        HP(pred_type='mixture'),
    )

    def default_tuning(
        *args,
        mixture_size=3,
        base_loss='nll',
        inhoc_grid=HP(method=None),
        posthoc_dataset='calib',
        posthoc_grid=HP(method=None)
    ):
        
        return Join(
            dist,
            *args,
            HP(mixture_size=mixture_size),
            HP(base_loss=base_loss),
            HP(inhoc_grid=inhoc_grid),
            HP(posthoc_dataset=posthoc_dataset),
            HP(posthoc_grid=posthoc_grid),
        )

    bs = [0.01, 0.05, 0.1, 0.2]

    inhoc_grid_list = [
        Join(HP(method='smooth_ecdf'), HP(alpha=alpha), HP(b=b))
        for alpha in [1]
        for b in bs
    ]

    posthoc_grid = HP(method=[None, 'smooth_ecdf'])

    return Union(
        default_tuning(posthoc_dataset=['calib', 'train'], base_loss='nll', posthoc_grid=posthoc_grid),
        default_tuning(HP(cal_size=[2048]), HP(inhoc_dataset=['batch']), base_loss='nll_inhoc_mc', inhoc_grid=inhoc_grid_list, posthoc_grid=posthoc_grid),
    )


def _get_tuning(config):
    if config.tuning_type == 'QRT':
        return get_tuning_for_QRT(config)
    elif config.tuning_type == 'QRT_per_epoch':
        return get_tuning_for_QRT_per_epoch(config)
    raise ValueError('Invalid tuning type')


def duplicates(choices):
    frozendict = lambda d: frozenset(d.items())
    frozen_choices = map(frozendict, choices)
    return [choice for choice, count in collections.Counter(frozen_choices).items() if count > 1]


def remove_duplicates(seq_of_dicts):
    seen = set()
    deduped_seq = []
    
    for d in seq_of_dicts:
        t = tuple(frozenset(d.items()))
        if t not in seen:
            seen.add(t)
            deduped_seq.append(d)
            
    return deduped_seq


def get_tuning(config):
    tuning = _get_tuning(config)
    tuning = remove_duplicates(tuning)
    dup = duplicates(tuning)
    assert len(dup) == 0, dup
    return tuning


if __name__ == '__main__':
    tuning = get_tuning_for_QRT(None)
    total = 0
    for t in tuning:
        total += len(list(t['posthoc_grid']))
    print(total)
