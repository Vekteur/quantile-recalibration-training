import logging
import shutil
from typing import List

from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.plugins.environments import SLURMEnvironment

from uq.utils.general import instantiate

log = logging.getLogger('uq')


class KeyboardInterruptCallback(Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            exit()


class DisableLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def __enter__(self):
        self.logger.propagate = False

    def __exit__(self, type, value, traceback):
        self.logger.propagate = True


def instantiate_callback(rc, cb_conf, monitor, process_position):
    if cb_conf.cls == TQDMProgressBar:
        cb_conf.args.process_position = process_position
    if cb_conf.cls in [EarlyStopping, ModelCheckpoint]:
        cb_conf.args.monitor = f'val/{monitor}'
    callback = instantiate(cb_conf)
    if cb_conf.cls == ModelCheckpoint:
        callback.dirpath = str(rc.checkpoints_path)
    return callback


def init_callbacks(rc, monitor, process_position):
    checkpoints_path = rc.checkpoints_path

    # Checkpoints in this directory must originate from a previous interrupted run.
    # They should be cleaned up.
    if rc.config.clean_previous:
        if checkpoints_path.exists():
            shutil.rmtree(checkpoints_path)
    callbacks = []
    if 'callbacks' in rc.config:
        for cb_conf in rc.config.callbacks.values():
            if cb_conf is None:
                continue
            callback = instantiate_callback(rc, cb_conf, monitor, process_position)
            if callback is not None:
                callbacks.append(callback)
    callbacks.append(KeyboardInterruptCallback())
    return callbacks


def init_loggers(rc):
    loggers = []
    if 'loggers' in rc.config:
        for lg_conf in rc.config.loggers.values():
            loggers.append(instantiate(lg_conf))
    return loggers


def init_trainer(rc, callbacks, loggers, **kwargs):
    trainer_name = 'default'
    if rc.hparams['base_model'] in ['nn', 'resnet']:
        trainer_name = 'lightning'
    with DisableLogger('pytorch_lightning.utilities.distributed'):
        trainer = instantiate(
            rc.config.trainer[trainer_name],
            callbacks=callbacks,
            logger=loggers,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            **kwargs,
        )
    return trainer


def init_trainer_with_loggers_and_callbacks(rc, model, process_position=0, **kwargs):
    # We set process_position at 0 for now because it is hard to setup with multiprocessing
    callbacks = init_callbacks(rc, model.monitor, process_position=process_position)
    loggers = init_loggers(rc)
    return init_trainer(rc, callbacks, loggers, **kwargs)


def get_model_hparams(rc, datamodule):
    total_size, input_size = datamodule.get_data_size()
    return {**rc.hparams, 'input_size': input_size}


def load_datamodule(rc):
    try:
        datamodule_config = rc.dataset_group_config.datamodule
    except ConfigAttributeError:
        log.error(f'The datamodule of {rc.dataset} is not present in the config')
        raise
    return instantiate(datamodule_config, rc=rc, name=rc.dataset, seed=2000 + rc.run_id)


def fit_with_profiling(rc, trainer, model, datamodule):
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function('training'):
            trainer.fit(model=model, datamodule=datamodule)
    prof.export_chrome_trace('tmp/trace.json')
    print(f'Profiling of {rc.summary_str()}', flush=True)
    print(
        prof.key_averages().table(sort_by='cpu_time_total', row_limit=50),
        flush=True,
    )


# import cProfile
# profiler = cProfile.Profile()


def train(rc, process_index):
    log.warn(f'Starting {rc.summary_str()}')
    # Init lightning datamodule
    # For each run index, different splits are selected.
    # An alternative would be to remove the randomness of the splits by fixing the same seed for each run index.
    datamodule = load_datamodule(rc)
    datamodule.load_datasets()   # Needed for inhoc_variant=="learned"

    # Init lightning model with the same seed for the same run index.
    if rc.tuning:
        seed = 1000 + rc.run_id
    else:
        seed = rc.run_id
    rc.seed = seed
    seed_everything(rc.seed)
    hps = get_model_hparams(rc, datamodule)
    model = rc.model_cls(rc=rc, hps=hps, datamodule=datamodule)
    trainer = init_trainer_with_loggers_and_callbacks(rc, model, process_position=process_index)

    # Training loop
    model.best_epoch_to_use = None
    trainer.fit(model=model, datamodule=datamodule)
    #fit_with_profiling(rc, trainer, model, datamodule)

    # profiler.print_stats()
    # print(flush=True)
    
    # Test the model
    for cb in trainer.callbacks:
        if type(cb) == ModelCheckpoint:
            best_model_path = cb.best_model_path
    assert best_model_path != ''
    hps = get_model_hparams(rc, datamodule)
    best_model = rc.model_cls.load_from_checkpoint(best_model_path, rc=rc, hps=hps, datamodule=datamodule)
    model.load_state_dict(best_model.state_dict())
    # This is needed to recompute the correct calibration map
    model.best_epoch_to_use = best_model.current_epoch
    # model.current_epoch = best_model.current_epoch

    model.trainer = trainer

    with DisableLogger('pytorch_lightning.utilities.distributed'):
        trainer.test(model=model, datamodule=datamodule, verbose=False)
        # We compute the validation on the best epoch after training because it is used for model selection
        # We only compute it if it has not been computed before and if the test metrics are computed
        if not rc.config.save_val_metrics and rc.config.save_test_metrics:
            rc.config.save_val_metrics = True
            trainer.validate(model=model, datamodule=datamodule, verbose=False)
            rc.config.save_val_metrics = False

    rcs = model.posthoc_manager.make_run_configs()
    log.warn(f'Finished {rc.summary_str()}')
    return rc, rcs
