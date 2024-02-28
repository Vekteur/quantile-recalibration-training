from uq.models.general.base_loss_module import BaseLossModule
from uq.models.general.nn_module import MixtureModule, SplineModule
from uq.models.posthoc.inhoc_module import InHocModule
from uq.models.regul.regul_module import (
    DistCDF_Regul,
    DistEntropySSRegul,
    DistEntropyMCRegul,
    DistQuantileRegul,
    NoRegul,
)


def get_pred_type_module(pred_type):
    return {
        'mixture': MixtureModule,
        'spline': SplineModule,
        # TODO: handle quantiles
    }[pred_type]


def get_regul_module(regul):
    return {
        None: NoRegul,
        'entropy_based_ss': DistEntropySSRegul,
        'entropy_based_mc': DistEntropyMCRegul,
        'cdf_based': DistCDF_Regul,
        'quantile_based': DistQuantileRegul,
        # TODO: handle others reguls
    }[regul]


def get_module(base_module, hparams):
    for key in ['base_model', 'pred_type']:
        assert key in hparams, (key, hparams)
    base_model = hparams['base_model']
    # TODO: handle NGBoost

    # pred_type
    pred_type = hparams['pred_type']
    module = get_pred_type_module(pred_type)(base_module, name='base')

    # regul
    regul = hparams.get('regul')
    module = get_regul_module(regul)(module, name='regul', metric_name='regul')

    # inhoc
    module = InHocModule(module, name='inhoc', datamodule=base_module.datamodule)

    # base_loss
    module = BaseLossModule(module, name='base_loss', metric_name='es_loss')
    return module
