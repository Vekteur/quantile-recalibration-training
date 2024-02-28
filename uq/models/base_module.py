from abc import abstractmethod
from typing import Sequence, Union
from timeit import default_timer

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback

from uq.metrics.posthoc_module_manager import PostHocModuleManager
from uq.models.general.build_model import get_module
from uq.utils.general import elapsed_timer, cycle
from uq.utils.hparams import HP


class BaseModule(LightningModule):
    def __init__(
        self,
        rc=None,
        hps=None,
        datamodule=None,
        **kwargs,
    ):
        super().__init__()
        self.rc = rc
        self.hps = hps
        self.datamodule = datamodule
        self.module = get_module(self, self.hps)
        self.capture_hparams(**self.hps)
        self.module.propagate_hparams(self.hps)
        self.module.build()
        self.started_stages = set() # To keep track of started stages to help with timing stages
        self.start_times = {} # To measure time per stage
        self.stage = 'predict' # To keep track of current stage to help with timing stages
        self.automatic_optimization = False # To measure backprop time


    def capture_hparams(
        self,
        lr=1e-3,
        posthoc_dataset='train',
        posthoc_grid=HP(method=None),
        batch_size=None,
        misspecification=None,
        interleaved=False,
        **kwargs,
    ):
        self.lr = lr
        self.posthoc_dataset = posthoc_dataset
        self.posthoc_grid = posthoc_grid
        self.batch_size = batch_size
        self.misspecification = misspecification
        self.interleaved = interleaved
        assert misspecification in [
            None,
            'small_mlp',
            'big_mlp',
            'homoscedasticity',
            'sharpness_reward',
        ]
        assert self.posthoc_dataset in ['train', 'calib']
        for posthoc_hparams in posthoc_grid:
            method = posthoc_hparams['method']
            assert method in [
                None,
                'ecdf',
                'stochastic_ecdf',
                'linear_ecdf',
                'smooth_ecdf',
                'conditional',
                'cqr',
            ]
        posthoc_grid_size = len(list(self.posthoc_grid))
        assert posthoc_grid_size >= 1
        self.posthoc_manager = PostHocModuleManager(self, self.posthoc_dataset, self.posthoc_grid)

    @property
    def monitor(self):
        return 'es_loss'

    @abstractmethod
    def build_model(self):
        pass

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def step(self, batch, batch_idx, stage):
        pass

    def timed_step(self, batch, batch_idx, stage):
        #print(f'Start step {stage}: {default_timer() - self.start_times[stage]:.3f}s', )
        self.stage = stage
        result = self.step(batch, batch_idx, stage)
        #print(f'End step {stage}: {default_timer() - self.start_times[stage]:.3f}s', )
        return result

    def predict_step(self, batch, batch_idx):
        return self(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        metrics = self.timed_step(batch, batch_idx, 'train')
        if self.rc.config.save_train_metrics:
            self.log(
                f'train/{self.monitor}',
                metrics[self.monitor],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.timed_step(batch, batch_idx, 'val')
        self.log(
            f'val/{self.monitor}',
            metrics[self.monitor],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return metrics

    def test_step(self, batch, batch_idx):
        return self.timed_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.parameters(), lr=self.lr)

    def configure_callbacks(self):
        self.module.before_train_starts()
        return super().configure_callbacks()

    def on_start(self, stage):
        self.stage = stage
        self.started_stages.add(stage)
        self.start_times[stage] = default_timer()
    
    def on_end(self, stage):
        self.stage = stage
        self.started_stages.remove(stage)
        self.advance_timer('time', default_timer() - self.start_times[stage])
        self.start_times.pop(stage)

    def on_train_start(self):
        self.scaler = self.trainer.datamodule.scaler_y
        self.on_start('train')

    def training_epoch_end(self, outputs):
        self.posthoc_manager.collect_per_step(outputs, 'train')
        # To eval metrics on the calibration set:
        # self.eval()
        # batch = self.trainer.datamodule.data_calib[:]
        # with torch.no_grad():
        #     output = self.timed_step(batch, 0, 'calib')
        # self.posthoc_manager.collect_per_step([output], 'calib')
        # self.train()

    def on_train_end(self):
        self.on_end('train')
        self.posthoc_manager.add_best_iter_metrics()
    
    def on_validation_start(self):
        self.on_start('val')

    def validation_epoch_end(self, outputs):
        self.posthoc_manager.collect_per_step(outputs, 'val')
    
    def on_validation_end(self):
        self.on_end('val')

    def on_test_start(self):
        self.scaler = self.trainer.datamodule.scaler_y
        self.on_start('test')

    def test_epoch_end(self, outputs):
        self.posthoc_manager.collect_per_step(outputs, 'test')
    
    def on_test_end(self):
        self.on_end('test')

    def advance_timer(self, name, amount, cancel_time=False, stage=None):
        if stage is None:
            stage = self.stage
        name = f'{stage}_{name}'
        self.posthoc_manager.advance_timer(name, amount, cancel_time=cancel_time)
