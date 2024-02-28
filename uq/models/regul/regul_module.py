from abc import abstractmethod

import torch

from ..decorator_module import DecoratorModule
from .marginal_regul import (
    cdf_based_regul_from_pit,
    sample_spacing_entropy_estimation,
    kde_mc_entropy_estimation,
    quantile_based_regul,
)
from uq.utils.general import cycle


class RegulModule(DecoratorModule):
    def __init__(self, *args, metric_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_name = metric_name

    def step(self, x, y, batch_idx, stage):
        pred = self.module.step(x, y, batch_idx, stage)
        self.loss = self.compute_regul_interleaved(pred, y)
        return pred

    def compute_regul_interleaved(self, pred, y):
        # We don't compute regul if interleaving is enabled and the current minibatch is the training dataset
        skip_regul = self.interleaved and self.base_module.current_epoch % 2 == 1
        if self.lambda_ != 0 and not skip_regul:
            regul = self.compute_regul(pred, y)
            loss = self.lambda_ * regul
        else:
            regul = torch.full([], torch.nan)
            loss = torch.full([], 0.)
        if self.metric_name is not None:
            self.metrics[self.metric_name] = regul.detach()
            self.metrics[f'{self.metric_name}_total'] = loss.detach()
        return loss

    @abstractmethod
    def compute_regul(self, dist, y):
        pass

    def capture_hparams(self, lambda_=0, interleaved=False, **kwargs):
        self.lambda_ = lambda_
        self.interleaved = interleaved
        return kwargs


class NoRegul(RegulModule):
    def compute_regul(self, dist, y):
        return torch.full([], 0.0)


class DistEntropySSRegul(RegulModule):
    def compute_regul(self, dist, y):
        pit = dist.cdf(y)
        return -sample_spacing_entropy_estimation(
            pit,
            spacing=self.spacing,
            neural_sort=self.neural_sort,
            s=self.s,
            divergence=self.divergence,
        )

    def capture_hparams(self, spacing=64, neural_sort=True, s=1., divergence='l2', **kwargs):
        super().capture_hparams(**kwargs)
        self.spacing = spacing
        self.neural_sort = neural_sort
        self.s = s
        assert divergence in ['l2', 'kl']
        self.divergence = divergence


class DistEntropyMCRegul(RegulModule):
    def compute_regul(self, dist, y):
        def eval_pit(batch):
            x, y = batch
            return self.module.predict(x).cdf(y.squeeze(1))

        if self.kde_dataiter is None:
            kde_pit = dist.cdf(y)
        else:
            kde_pit = eval_pit(next(self.kde_dataiter))
        
        if self.mc_dataiter is self.kde_dataiter:
            mc_pit = kde_pit
        elif self.mc_dataiter is None:
            mc_pit = dist.cdf(y)
        else:
            mc_pit = eval_pit(next(self.mc_dataiter))

        return -kde_mc_entropy_estimation(
            kde_pit,
            mc_pit,
            b=self.b,
            base_module=self.base_module,
        )

    def capture_hparams(self, b=0.1, kde_dataset='batch', mc_dataset='batch', cal_size=2048, **kwargs):
        super().capture_hparams(**kwargs)
        self.kde_dataiter = None
        if kde_dataset != 'batch':
            self.kde_dataiter = iter(cycle(self.base_module.datamodule.cal_dataloader(kde_dataset, cal_size)))
        self.mc_dataiter = None
        if mc_dataset == kde_dataset:
            self.mc_dataiter = self.kde_dataiter
        elif mc_dataset != 'batch':
            self.mc_dataiter = iter(cycle(self.base_module.datamodule.cal_dataloader(mc_dataset, cal_size)))
        self.b = b


class DistCDF_Regul(RegulModule):
    def compute_regul(self, dist, y):
        return cdf_based_regul_from_pit(dist.cdf(y), self.L, s=self.s)

    def capture_hparams(self, L=2, s=100, **kwargs):
        super().capture_hparams(**kwargs)
        self.L = L
        self.s = s


class DistQuantileRegul(RegulModule):
    def compute_regul(self, dist, y):
        return quantile_based_regul(dist.cdf(y), self.L, neural_sort=self.neural_sort, s=self.s)

    def capture_hparams(self, L=2, neural_sort=False, s=0.1, **kwargs):
        super().capture_hparams(**kwargs)
        self.L = L
        self.neural_sort = neural_sort
        self.s = s
