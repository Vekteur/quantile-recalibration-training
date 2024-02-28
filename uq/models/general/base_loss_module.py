import torch

from uq.metrics.general import nll
from uq.utils.hparams import HP
from uq.utils.general import elapsed_timer

from ..decorator_module import DecoratorModule

#from uq.train import profiler


class BaseLossModule(DecoratorModule):
    def __init__(self, *args, coeff=1, metric_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeff = coeff
        self.metric_name = metric_name

    def step(self, x, y, batch_idx, stage):
        pred = self.module.step(x, y, batch_idx, stage)
        with elapsed_timer() as time:
            self.loss = self.compute_base_loss_interleaved(pred, y)
        self.base_module.advance_timer('loss_time', time())
        assert self.loss != torch.inf   # A loss of infinity causes an exception
        return pred

    def compute_base_loss(self, dist, y):
        if self.base_loss == 'nll':
            loss = nll(dist, y).mean()
        elif self.base_loss == 'crps':
            loss = dist.crps(y).mean()
        elif self.base_loss == 'nll_inhoc_mc':
            loss = -dist.log_prob_regul_mc(y).mean()
        elif self.base_loss == 'nll_inhoc_ss':
            loss = -dist.log_prob_regul_ss(y).mean()
        else:
            raise ValueError(f'Invalid base_loss: {self.hparams.base_loss}')
        if self.misspecification == 'sharpness_reward':
            if self.pred_type == 'mixture':
                loss = loss + dist.stddev.mean() * 10.0
            else:
                raise NotImplementedError('The standard deviation of a spline is not implemented')
        return loss

    def compute_base_loss_interleaved(self, pred, y):
        # We don't compute the base loss if interleaving is enabled and the current minibatch is the interleaved dataset
        skip_base_loss = self.interleaved and self.base_module.current_epoch % 2 == 0
        if not skip_base_loss:
            base_loss = self.compute_base_loss(pred, y)
            loss = self.coeff * base_loss
        else:
            base_loss = torch.full([], torch.nan)
            loss = torch.full([], 0)
        if self.metric_name is not None:
            self.metrics[self.metric_name] = base_loss.detach()
            if hasattr(pred, 'metrics'):
                assert isinstance(pred.metrics, dict)
                self.metrics = {**self.metrics, **pred.metrics}
                pred.metrics = {}
        return loss

    def capture_hparams(
        self,
        base_loss=None,
        misspecification=None,
        pred_type=None,
        interleaved=False,
        inhoc_grid=HP(method=None),
        **kwargs,
    ):
        self.base_loss = base_loss
        self.misspecification = misspecification
        self.pred_type = pred_type
        self.interleaved = interleaved
        inhoc_hparams = next(iter(inhoc_grid))
        if 'lambda_' in inhoc_hparams and self.metric_name == 'base_loss':
            self.base_loss = 'nll_lambda_'
