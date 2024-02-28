import logging
import pickle
import shutil
import traceback
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import as_completed, get_client
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from uq.models.dist.base_dist_module import DistModule
from uq.train import train
from uq.tuning import get_tuning
from uq.utils.run_config import RunConfig
from uq.datamodules.uci.download_uci import download_all_uci
from uq.datamodules.openml.download_openml import download_openml_suite

log = logging.getLogger('uq')


def get_best_hparams(results):
    df = pd.DataFrame(
        {
            'hparams': map(lambda rc: rc.hparams, results),
            # `hparams_id` is needed because groupby needs a hashable type
            'hparams_id': map(lambda rc: tuple(rc.hparams.items()), results),
            'score': map(lambda rc: rc.metrics['best_score'], results),
        }
    )
    assert len(df) > 0
    df = df.groupby('hparams_id').agg({'score': 'mean', 'hparams': lambda x: x.values[0]})
    best_score = df['score'].min()
    try:
        best_hparams = df.query(f'score == {best_score}')['hparams'].iloc[0]
    except KeyError:
        log.warn('The best score is nan')
        log.warn(df, flush=True)
        best_hparams = df['hparams'].iloc[0]
    return best_hparams


def mute_cumbersome_logging():
    logging.getLogger('lightning_lite.utilities.seed').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning.utilities.rank_zero').setLevel(logging.WARNING)
    # logging.getLogger('torch.distributed.nn.jit.instantiator').setLevel(logging.WARNING)
    logging.getLogger('distributed.diskutils').setLevel(logging.WARN)
    warnings.filterwarnings('ignore', '.*Unmanaged memory use is high.*')
    warnings.filterwarnings(
        'ignore',
        r'.*The `srun` command is available on your system but is not used.*',
        category=PossibleUserWarning,
    )
    warnings.filterwarnings(
        'ignore',
        r'.*GPU available but not used\. Set `accelerator` and `devices` using.*',
        category=PossibleUserWarning,
    )
    warnings.filterwarnings(
        'ignore',
        r'.*does not have many workers which may be a bottleneck\. Consider increasing the value of the.*',
        category=PossibleUserWarning,
    )


def train_and_save(rc, hparams, pm):
    logging.basicConfig(level=logging.WARN)
    mute_cumbersome_logging()

    rc.hparams = hparams
    if rc.storage_path.exists():
        # # Workaround to overwrite results with specific hyperparameters
        # if rc.hparams['lambda_'] == 0:
        #     pass # We overwrite the results
        # else:
        with open(rc.storage_path, 'rb') as f:
            return pickle.load(f)

    index = 0 if pm is None else pm.request().result()
    # Run training
    rc, rcs = train(rc, index)

    if pm is not None:
        pm.free(index).result()

    rc.storage_path.parent.mkdir(parents=True, exist_ok=True)
    if rc.config.remove_checkpoints:
        assert len(list(rc.checkpoints_path.rglob('*'))) <= 2, list(rc.checkpoints_path.rglob('*'))
        # In case of error, check that I am not running the same runs (with same hyperparameters) concurrently!
        shutil.rmtree(rc.checkpoints_path)
    with open(rc.storage_path, 'wb') as f:
        # Don't save the whole config. This should save a lot of space.
        # However, it should be readded after loading the RunConfig again.
        for rc_posthoc in rcs:
            rc_posthoc.config = None
            if 'inhoc_grid' in rc_posthoc.hparams:
                inhoc_grid = rc_posthoc.hparams['inhoc_grid']
                inhoc_hparams = next(iter(inhoc_grid))
                hparams_with_prefix = {f'inhoc_{key}': value for key, value in inhoc_hparams.items()}
                rc_posthoc.hparams.update(hparams_with_prefix)
        pickle.dump(rcs, f)
    return rc


class PositionManager:
    def __init__(self, size):
        self.slots = [False for _ in range(size)]

    def free(self, i):
        self.slots[i] = False

    def request(self):
        for i, slot in enumerate(self.slots):
            if not slot:
                self.slots[i] = True
                return i
        log.warn('No slot available')
        return 0


class Runner:
    def __init__(self, config, manager='sequential'):
        self.config = config
        assert manager in ['sequential', 'dask', 'joblib']
        self.manager = manager
        self.tasks = []
        if self.manager == 'dask':
            pm_future = self.submit(
                PositionManager,
                self.config.nb_workers,
                actor=True,
            )
            self.pm = pm_future.result()
        else:
            self.pm = None
    
    def submit(self, fn, *args, priority=None, **kwargs):
        if self.manager == 'dask':
            return get_client().submit(fn, *args, **kwargs, priority=priority)
        elif self.manager == 'joblib':
            return (fn, args, kwargs)
        else:
            return fn(*args, **kwargs)

    def train_in_parallel(self, rc, hparams, priority):
        return self.submit(
            train_and_save,
            rc,
            hparams,
            self.pm,
            priority=priority,
        )

    def grid_search(self, rc, priority):
        grid = get_tuning(rc.config)
        for hparams in grid:
            rc.model_cls = DistModule
            future_rc = self.train_in_parallel(rc, hparams, priority)
            self.tasks.append(future_rc)

    def run_tuning(self, rc, priority):
        rc.tuning = True
        from uq.utils.general import print_once
        for run_id in range(self.config.repeat_tuning):
            new_rc = copy(rc)
            new_rc.run_id = run_id
            self.grid_search(new_rc, priority)

    def close(self):
        if self.manager == 'dask':
            for future in as_completed(self.tasks):
                if future.status == 'error':
                    log.warn('Error in parallel task')
                    print('=' * 60)
                    print('Traceback')
                    print('=' * 60)
                    traceback.print_tb(future.traceback())
                    print('Exception:', future.exception())
        elif self.manager == 'joblib':
            Parallel(n_jobs=self.config.nb_workers)(delayed(wrapped_fn)(fn, *args, **kwargs) for fn, args, kwargs in self.tasks)


def wrapped_fn(fn, *args, **kwargs):
    # This function is used to catch exceptions in parallel tasks without stopping the other tasks
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print('=== Start of error in wrapped function ===', flush=True)
        print('args:', flush=True)
        for arg in args:
            if type(arg) == RunConfig:
                print(arg.summary_str(bold=False))
            else:
                print(arg)
        print('kwargs:', flush=True)
        print(kwargs)
        print('Traceback:', flush=True)
        traceback.print_exc()
        print('=== End of error in wrapped function ===', flush=True)
        # We do not raise the exception here to avoid stopping the other tasks
        #raise e


def run_all(config: DictConfig, manager='sequential'):
    logging.basicConfig(level=logging.WARN)
    log.setLevel(logging.WARN)
    OmegaConf.save(config, Path(config.log_dir) / 'config.yaml')

    # We download the datasets here before training
    # Note that they will be downloaded again during training if a specific dataset is not found
    if not Path('data').exists():
        data_path = Path(config.data_dir)
        for suite_id in [269, 297, 299]:
            download_openml_suite(suite_id, data_path)
        download_all_uci(data_path)

    runner = Runner(config, manager=manager)
    priority = 0
    for dataset_group, dataset_group_config in config.dataset_groups.items():
        # log.info(f"Dataset group: \033[1;4m{dataset_group}\033[0m")
        for dataset in dataset_group_config.names:
            # log.info(f"  Dataset: \033[4m{dataset}\033[0m")
            rc = RunConfig(
                config=config,
                dataset_group=dataset_group,
                dataset=dataset,
            )
            runner.run_tuning(rc, priority)
            priority -= 1
    runner.close()
