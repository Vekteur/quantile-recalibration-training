import torch
from torch.distributions import (
    AffineTransform,
    CumulativeDistributionTransform,
)

from uq.models.pred_type.ecdf import EmpiricalCDF, StochasticEmpiricalCDF
from uq.models.pred_type.linear_spline import LinearSpline
from uq.models.pred_type.mixture_dist import LogisticMixtureDist
from uq.models.pred_type.reflected_dist import ReflectedDist
from uq.models.pred_type.truncated_dist import TruncatedDist

PostHocPitCalibration = EmpiricalCDF
PostHocStochasticPitCalibration = StochasticEmpiricalCDF


class PostHocLinearPitCalibration(LinearSpline):
    def __init__(self, pit, **kwargs):
        bx = torch.sort(pit)[0]
        # centered_bins(len(bx))
        by = (torch.arange(len(bx)) + 1) / (len(bx) + 1)
        super().__init__(bx, by, **kwargs)


class SmoothEmpiricalCDF(CumulativeDistributionTransform):
    def __init__(
        self,
        x,
        b=0.1,
        reflected=True,
        truncated=False,
        epoch=None,
        batch_idx=None,
        base_module=None,
        **kwargs,
    ):
        self.epoch = epoch
        self.batch_idx = batch_idx
        assert x.dim() == 1
        N = x.shape[0]
        dist = LogisticMixtureDist(x, torch.tensor(b * N ** (-1 / 5)), probs=torch.ones_like(x), base_module=base_module)
        if reflected:
            dist = ReflectedDist(dist, torch.tensor(0.0), torch.tensor(1.0))
        if truncated:
            dist = TruncatedDist(dist, torch.tensor(0.0), torch.tensor(1.0))
        super().__init__(dist, **kwargs)


PostHocSmoothPitCalibration = SmoothEmpiricalCDF


def build_recalibration_map(pit, posthoc_method, posthoc_hparams, epoch=None, batch_idx=None, base_module=None):
    if posthoc_method == 'ecdf':
        return PostHocPitCalibration(pit)
    elif posthoc_method == 'stochastic_ecdf':
        return PostHocStochasticPitCalibration(pit)
    elif posthoc_method == 'linear_ecdf':
        return PostHocLinearPitCalibration(pit)
    elif posthoc_method == 'smooth_ecdf':
        return PostHocSmoothPitCalibration(pit, epoch=epoch, batch_idx=batch_idx, base_module=base_module, **posthoc_hparams)
    else:
        raise ValueError()
