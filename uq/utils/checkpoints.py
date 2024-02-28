import json
import logging

from uq.utils.run_config import RunConfig
from uq.train import load_datamodule, get_model_hparams

log = logging.getLogger('uq')


def get_hparams(rc, config):
    if config.tuning:   # If there was hyperparameter tuning before the run
        hparams_path = rc.model_path / 'best_hparams.json'
        with open(hparams_path, 'r') as f:
            hparams = json.load(f)
    else:
        hparams = {}
    return hparams


def get_checkpoint_path(rc, epoch=None):
    if epoch == 'last':
        ckpt_name = 'last.ckpt'
    elif epoch == 'best':
        ckpts_list = []
        for ckpt_path in rc.checkpoints_path.iterdir():
            if ckpt_path.name != 'last.ckpt':
                ckpts_list.append(ckpt_path.name)
        ckpts_list.sort(key=lambda x: int(x[6:10]))
        if len(ckpts_list) != 1:
            log.warn(f'More than 1 checkpoint available at {rc.checkpoints_path} ({ckpts_list})')
        ckpt_name = ckpts_list[-1]
    else:
        ckpt_name = f'epoch_{epoch:04d}.ckpt'
    return rc.checkpoints_path / ckpt_name


def load_model_checkpoint(rc, datamodule, epoch='best'):
    hps = get_model_hparams(rc, datamodule)
    checkpoint_path = get_checkpoint_path(rc, epoch=epoch)
    return rc.model_cls.load_from_checkpoint(checkpoint_path, rc=rc, hps=hps, datamodule=datamodule)


def load_rc_checkpoint(config, dataset_group, dataset, run_id=0, hparams=None, model_cls=None):
    # We suppose that, if hyperparameters are given, we want models that were obtained during
    # hyperparameter tuning with specific hyperparameters.
    tuning = hparams is not None
    rc = RunConfig(
        config=config,
        dataset_group=dataset_group,
        dataset=dataset,
        run_id=run_id,
        tuning=tuning,
        hparams=hparams,
        model_cls=model_cls,
    )
    if not rc.tuning:   # If the run was not obtained during hyperparameter tuning
        rc.hparams = get_hparams(config, rc)
    return rc
