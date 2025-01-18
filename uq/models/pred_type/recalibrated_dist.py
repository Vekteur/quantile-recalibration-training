import torch
from torch.distributions import Normal, TransformedDistribution, Uniform
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    CumulativeDistributionTransform,
)
from torch.distributions.utils import _sum_rightmost

from uq.utils.dist import adjust_unit_tensor
from uq.models.regul.marginal_regul import sample_spacing_entropy_estimation


class UnitUniform(Uniform):
    def __init__(self, batch_shape, *args, **kwargs):
        super().__init__(torch.zeros(batch_shape), torch.ones(batch_shape), *args, **kwargs)

    def log_prob(self, value):
        value = adjust_unit_tensor(value)
        eps = 1e-7
        value[value == 0.0] += eps
        value[value == 1.0] -= eps
        return super().log_prob(value)


class RecalibratedDist(TransformedDistribution):
    def __init__(self, dist, posthoc_model, scaler=None, alpha=1.0, spacing=64, neural_sort=True, s=1.0):
        self.dist = dist
        self.posthoc_model = posthoc_model
        # Regularization parameter (always equal to 1 during log_prob evaluation)
        self.alpha = alpha
        self.spacing = spacing
        self.neural_sort = neural_sort
        self.s = s

        base_dist = UnitUniform(dist.batch_shape)
        transforms = [
            self.posthoc_model.inv,
            CumulativeDistributionTransform(dist).inv,
        ]
        if scaler is not None:
            transforms.append(AffineTransform(scaler.mean_, scaler.scale_))

        self.metrics = {} # Regularization metrics that will be captured by the inhoc_module

        super().__init__(base_dist, transforms)

    def crps(self, value):
        raise NotImplementedError()

    def unnormalize(self, scaler):
        return RecalibratedDist(self.dist, self.posthoc_model, scaler, self.alpha)

    def log_prob_regul_mc(self, value):
        # When alpha = 1, this function returns the same value as self.log_prob(value).
        # The second term computes the entropy of the PIT using Monte Carlo estimation.
        dist_log_prob = self.dist.log_prob(value)
        pit = self.dist.cdf(value)
        neg_entropy = self.posthoc_model.log_abs_det_jacobian(pit, None)
        # We add minus signs before the metrics to take into account the negation of the NLL
        self.metrics['inhoc_base_loss'] = -dist_log_prob.mean().detach()
        self.metrics['inhoc_regul'] = -neg_entropy.mean().detach()
        neg_entropy_scaled = self.alpha * neg_entropy
        self.metrics['inhoc_regul_scaled'] = -neg_entropy_scaled.mean().detach()
        return dist_log_prob + neg_entropy_scaled

    def log_prob_regul_ss(self, value):
        # The second term computes the entropy of the PIT using sample-spacing estimation.
        dist_log_prob = self.dist.log_prob(value)
        pit = self.dist.cdf(value)
        neg_entropy = -sample_spacing_entropy_estimation(
            pit,
            spacing=self.spacing,
            neural_sort=self.neural_sort,
            s=self.s,
        )
        self.metrics['inhoc_base_loss'] = -dist_log_prob.mean().detach()
        self.metrics['inhoc_regul'] = -neg_entropy.mean().detach()
        neg_entropy_scaled = self.alpha * neg_entropy
        self.metrics['inhoc_regul_scaled'] = -neg_entropy_scaled.mean().detach()
        return dist_log_prob + neg_entropy_scaled


if __name__ == '__main__':
    from uq.models.posthoc.recalibration import SmoothEmpiricalCDF

    torch.set_printoptions(sci_mode=False)
    torch.manual_seed(0)

    dist = Normal(0, 1)
    y = torch.linspace(-3, 3, 1024)
    pit = dist.cdf(y)

    posthoc_model = SmoothEmpiricalCDF(pit)
    dist = RecalibratedDist(dist, posthoc_model)
    assert (dist.log_prob(y) == dist.log_prob_regul_mc(y)).all()
    dist.log_prob_regul_mc(y)
    print(dist.metrics['inhoc_regul'])
    dist.log_prob_regul_ss(y)
    print(dist.metrics['inhoc_regul'])

