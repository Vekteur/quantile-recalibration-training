from abc import abstractmethod

import torch
from torch.utils.data import TensorDataset
from torch.nn import Parameter

from uq.utils.general import elapsed_timer, cycle

from ..pred_type.recalibrated_dist import RecalibratedDist
from .recalibration import build_recalibration_map


# def get_posthoc_data(datamodule, posthoc_dataset, posthoc_method, data=None):
#     def sample(data):
#         x, y = data[:]
#         max_size = 2048
#         if y.shape[0] <= max_size:
#             return x, y
#         idx = torch.multinomial(torch.ones(y.shape[0]), max_size, replacement=False)
#         return x[idx], y[idx]

#     # When posthoc_dataset == 'batch', the post-hoc model is computed on half the batch during training
#     # but on the calibration dataset during validation and testing
#     if posthoc_dataset in ['calib', 'batch']:
#         data = sample(datamodule.data_calib)
#     elif posthoc_dataset == 'train':
#         data = sample(datamodule.data_train)
#     else:
#         # Else, the dataset is directly given as argument
#         data = sample(data)
#     return data


def get_posthoc_transformer(base_module, module, dataset, hparams, variant=None):
    method = hparams['method']
    if method is None:
        return IdentityTransformer(base_module, module, dataset, hparams, variant)
    elif method in ['ecdf', 'stochastic_ecdf', 'linear_ecdf', 'smooth_ecdf']:
        return RecalibratedDistTransformer(base_module, module, dataset, hparams, variant)
    raise ValueError('Invalid posthoc method:', method)


class PosthocTransformer:
    def __init__(self, base_module, module, dataset, hparams, variant=None):
        from ..base_module import BaseModule
        from ..decorator_module import ComponentModule

        #assert isinstance(base_module, BaseModule)
        self.base_module = base_module
        #assert isinstance(module, ComponentModule)
        self.module = module
        self.dataset = dataset
        self.hparams = hparams.copy()
        assert 'method' in self.hparams
        self.method = self.hparams.pop('method')
        self.mc_dataset = self.hparams.pop('mc_dataset', 'batch')
        self.alpha = self.hparams.pop('alpha', 1.0)
        self.cal_size = self.hparams.pop('cal_size', 2048)
        self.variant = variant
        self.epoch, self.batch_idx = None, None

        self.init_dataiters()
    
    def init_dataiters(self):
        # We initialize the dataloaders for creating the calibration maps and optional Monte Carlo evaluation
        self.dataiter = None
        # It is extremely important to not use the current batch during validation and testing
        # to compute the posthoc model because it would lead to data leak.
        # Instead, we use the training dataset.
        cal_dataset = 'train' if self.dataset == 'batch' else self.dataset
        self.dataiter = iter(cycle(self.base_module.datamodule.cal_dataloader(cal_dataset, self.cal_size)))
        self.mc_dataiter = None
        if self.mc_dataset == self.dataset:
            self.mc_dataiter = self.dataiter
        elif self.mc_dataset != 'batch':
            self.mc_dataiter = iter(cycle(self.base_module.datamodule.cal_dataloader(self.mc_dataset, self.cal_size)))

        # We initialize the parameters to learn with the correct size if the variant is learned
        self.params = None
        if self.variant == 'learned':
            if self.dataset == 'batch':
                n_params = self.base_module.batch_size
                assert n_params is not None
            else:
                x, y = next(self.dataiter)
                n_params = y.shape[0]
            self.params = Parameter(torch.rand(n_params))

    @abstractmethod
    def __call__(self, dist):
        pass

    def build(self, epoch, batch_idx, stage, batch=None):
        # The posthoc/inhoc model has to be updated once at validation and test time
        # because dropout could change the predictions of the base model.
        # This could have a negative impact on the performance of the posthoc model.
        # We set `batch_idx` to -1 so that the posthoc_model is always updated during validation
        # and test time, and the fixed value ensures that the posthoc_model is not recomputed.
        if stage != 'train':
            batch_idx = -1
        
        # We do not recompute the posthoc model if nothing has changed.
        if (self.epoch, self.batch_idx) == (epoch, batch_idx):
            return
        if self.variant == 'only_init' and (epoch, batch_idx) != (0, 0):
            return
        
        self.epoch, self.batch_idx = epoch, batch_idx

        with elapsed_timer() as time:
            if self.dataset == 'batch' and stage == 'train':
                x, y = batch
            else: # This includes the case when self.dataset == 'batch' and stage != 'train'
                x, y = next(self.dataiter)
        self.base_module.advance_timer('posthoc_data_time', time())
        if self.variant in ['only_init', 'no_grad']:
            with torch.no_grad():
                self._build(x, y, epoch, batch_idx)
        else:
            self._build(x, y, epoch, batch_idx)

    @abstractmethod
    def _build(self, x, y, epoch, batch_idx):
        pass


class IdentityTransformer(PosthocTransformer):
    def __call__(self, dist):
        return dist

    def build(self, epoch, batch_idx, stage, batch=None):
        pass


class RecalibratedDistTransformer(PosthocTransformer):
    def __call__(self, dist):
        return RecalibratedDist(dist, self.recalibration_map, alpha=self.alpha)

    def _build(self, x, y, epoch, batch_idx):
        if self.variant == 'learned':
            pit = self.params
        else:
            with elapsed_timer() as time:
                pit = self.module.predict(x).cdf(y.squeeze(dim=-1))
            self.base_module.advance_timer('pits_time', time())
        self.pit = pit
        self.recalibration_map = build_recalibration_map(
            pit, self.method, self.hparams, epoch=epoch, batch_idx=batch_idx, base_module=self.base_module
        )
