import torch
import torch.nn.functional as F
from pyro.distributions import Logistic
from torch.distributions import Categorical, MixtureSameFamily, Normal

from uq.metrics.quantiles import crps_normal_mixture
from uq.utils.dist import icdf_from_cdf
from uq.utils.general import elapsed_timer


# Mixture distribution for a location-scale family
class MixtureDist(MixtureSameFamily):
    def __init__(self, component_dist_class, means, stds, *, probs=None, logits=None):
        mix_dist = Categorical(probs=probs, logits=logits)
        self.component_dist_class = component_dist_class
        component_dist = self.component_dist_class(means, stds)
        super().__init__(mix_dist, component_dist)

    def icdf(self, value):
        return icdf_from_cdf(self, value)

    def affine_transform(self, loc, scale):
        """
        Let $X ~ Dist(\mu, \sigma)$. Then $a + bX ~ Dist(a + b \mu, b \sigma)$.
        The reasoning is similar for a mixture.
        """
        component_dist = self.component_distribution
        mix_dist = self.mixture_distribution
        means, stds = component_dist.loc, component_dist.scale
        means = loc + scale * means
        stds = scale * stds
        return type(self)(means, stds, logits=mix_dist.logits)

    def unnormalize(self, scaler):
        return self.affine_transform(scaler.mean_, scaler.scale_)

    def normalize(self, scaler):
        return self.affine_transform(-scaler.mean_ / scaler.scale_, 1.0 / scaler.scale_)

    def rsample(self, sample_shape=torch.Size(), tau=1):
        """
        Returns:
            Tensor: A tensor of shape `[batch_size, n_samples]`
        """
        raise NotImplementedError()


class NormalMixtureDist(MixtureDist):
    def __init__(self, *args, **kwargs):
        super().__init__(Normal, *args, **kwargs)

    def crps(self, value):
        return crps_normal_mixture(self, value)


class LogisticMixtureDist(MixtureDist):
    def __init__(self, *args, base_module=None, **kwargs):
        super().__init__(Logistic, *args, **kwargs)
        self.base_module = base_module

    def log_sigmoid(x):
        return -torch.where(x > 0, torch.log(1 + torch.exp(-x)), x + torch.log(1 + torch.exp(-x)))
    
    def log_prob(self, x):
        with elapsed_timer() as time:
            res = super().log_prob(x)
        if self.base_module is not None:
            self.base_module.advance_timer('logistic_log_prob_time', time())
        return res
