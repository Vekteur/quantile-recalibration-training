import torch
from torch import nn
from torch.distributions import (
    Distribution,
    constraints,
    Uniform,
    MixtureSameFamily,
    Categorical,
    TransformedDistribution,
    CumulativeDistributionTransform,
)
from pyro.distributions import Logistic


class Recalibrator(nn.Module):
    def __init__(self, pit, b=0.1):
        super().__init__()
        self.recalibration_map = SmoothEmpiricalCDF(pit, b=b)

    def __call__(self, dist):
        return RecalibratedDist(dist, self.recalibration_map)


class SmoothEmpiricalCDF(CumulativeDistributionTransform):
    def __init__(self, x, b=0.1, **kwargs):
        assert x.dim() == 1
        N = x.shape[0]
        dist = MixtureSameFamily(
            Categorical(probs=torch.ones_like(x)), 
            Logistic(x, torch.tensor(b * N ** (-1 / 5)))
        )
        dist = ReflectedDist(dist, torch.tensor(0.0), torch.tensor(1.0))
        super().__init__(dist, **kwargs)


# We need to define a custom Uniform distribution because the default one does not allow values of 0 and 1
class UnitUniform(Uniform):
    def __init__(self, batch_shape, *args, **kwargs):
        super().__init__(
            torch.zeros(batch_shape), torch.ones(batch_shape), *args, **kwargs
        )

    def log_prob(self, value):
        # Workaround because values of 0 and 1 are not allowed
        eps = 1e-7
        value[value == 0.0] += eps
        value[value == 1.0] -= eps
        return super().log_prob(value)


class RecalibratedDist(TransformedDistribution):
    def __init__(self, dist, posthoc_model):
        base_dist = UnitUniform(dist.batch_shape)
        transforms = [
            posthoc_model.inv,
            CumulativeDistributionTransform(dist).inv,
        ]
        super().__init__(base_dist, transforms)


class ReflectedDist(Distribution):
    support = constraints.real
    has_rsample = False

    def __init__(self, dist, a=-torch.inf, b=torch.inf):
        self.dist = dist
        self.a = a
        self.b = b

    @property
    def batch_shape(self):
        return self.dist._batch_shape

    def cdf(self, value):
        vb = self.dist.cdf(2 * self.b - value)
        va = self.dist.cdf(2 * self.a - value)
        v = self.dist.cdf(value)
        res = 1 - vb + v - va
        assert (
            (vb <= 1).all()
            and (0 <= va).all()
            and (0 <= v).all()
            and (v <= 1).all()
        ), f'{vb.max()}, {va.min()}, {v.min()}, {v.max()}'
        res[value < self.a] = 0
        res[self.b < value] = 1
        assert (0 <= res).all() and (
            res <= 1
        ).all(), f'{res.min()}, {res.max()}'
        return res

    def log_prob(self, value):
        log_probs = torch.stack(
            [
                self.dist.log_prob(2 * self.b - value),
                self.dist.log_prob(value),
                self.dist.log_prob(2 * self.a - value),
            ],
            dim=-1,
        )
        res = torch.logsumexp(log_probs, dim=-1)
        # clone is needed to avoid in-place operation
        res = res.clone()
        res[value < self.a] = -torch.inf
        res[self.b < value] = -torch.inf
        return res

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.dist.batch_shape
        rand = torch.rand(shape, device=self.a.device)
        return self.icdf(rand)
