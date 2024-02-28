import logging

import torch

from uq.models.general.base_loss_module import BaseLossModule
from uq.models.posthoc.posthoc_model import get_posthoc_transformer
from uq.utils.hparams import HP
from uq.utils.general import elapsed_timer

from ..decorator_module import DecoratorModule

log = logging.getLogger('uq')


class InHocModule(DecoratorModule):
    def __init__(self, *args, datamodule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.datamodule = datamodule

    def predict(self, x):
        return self.inhoc_model(self.module.predict(x))

    def step(self, x, y, batch_idx, stage):
        log.debug(f'epoch: {self.base_module.current_epoch}, idx: {batch_idx}')
        # Compute inhoc model
        with elapsed_timer() as time:
            # We can't assign to current_epoch directly so we had to use this workaround.
            epoch = self.base_module.best_epoch_to_use
            if epoch is None:
                epoch = self.base_module.current_epoch
            # print(f'Building {type(self.inhoc_model)} during {self.base_module.stage}', flush=True)
            # for module in self.base_module.posthoc_manager.modules:
            #     print([metric for metric in module.collector.metrics if type(metric) == str])
            self.inhoc_model.build(epoch, batch_idx, stage, batch=(x, y))
        self.base_module.advance_timer('build_time', time())
        pred = self.module.step(x, y, batch_idx, stage)
        # No need to time this step
        pred = self.inhoc_model(pred)
        return pred

    def capture_hparams(
        self,
        inhoc_dataset='batch',
        inhoc_grid=HP(method=None),
        inhoc_variant=None,
        **kwargs,
    ):
        self.dataset = inhoc_dataset
        assert self.dataset in ['train', 'calib', 'batch']
        self.inhoc_variant = inhoc_variant
        assert self.inhoc_variant in [None, 'only_init', 'no_grad', 'learned']
        assert len(list(inhoc_grid)) == 1
        self.inhoc_hparams = next(iter(inhoc_grid))

        self.inhoc_model = get_posthoc_transformer(
            self.base_module,
            self.module,
            self.dataset,
            self.inhoc_hparams,
            self.inhoc_variant,
        )
