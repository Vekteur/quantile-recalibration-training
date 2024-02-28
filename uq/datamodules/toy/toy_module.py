import numpy as np
import torch
import torch.distributions as D

from ..base_datamodule import BaseDataModule


def toy_cond_dist_simple(x):
    return D.Normal(torch.zeros_like(x), torch.ones_like(x))


def toy_cond_dist_sine(x):
    mean = 2 * (x + torch.sin(4 * x) + torch.sin(13 * x))
    std = torch.sqrt((x * 10 + 1) / 10)
    dist = D.Normal(mean, std)
    return dist


class CorrectExponential(D.Exponential):
    def log_prob(self, value):
        unsafe_log_prob = super().log_prob(value)
        return torch.where(
            self.support.check(value),
            unsafe_log_prob,
            torch.full_like(unsafe_log_prob, -torch.inf),
        )


def toy_cond_dist_rain(x):
    # Probability of rain from 1 to 0.2
    rate1 = torch.full_like(x, 0.5)   # There is rain
    rate2 = torch.full_like(x, 20)   # There is almost no rain
    rates = torch.stack([rate1, rate2], dim=-1)
    prob_rain = 0.8 - 0.6 * x
    probs = torch.stack([prob_rain, 1 - prob_rain], dim=-1)
    mix = D.Categorical(probs)
    comp = CorrectExponential(rates, validate_args=False)
    dist = D.MixtureSameFamily(mix, comp, validate_args=False)
    return dist


def toy_cond_dist_exp_varying_rate(x):
    rate = 0.5 + 2 * x
    dist = CorrectExponential(rate, validate_args=False)
    return dist


def toy_cond_dist_trimodal(x):
    mu1 = 5.0 - 10.0 * x
    mu2 = torch.full_like(mu1, 5.0)
    mu3 = torch.full_like(mu1, -5.0)
    mus = torch.stack([mu1, mu2, mu3], dim=-1)
    stds = torch.ones_like(mus)
    probs = torch.ones(3)[None, :].repeat(x.shape[0], 1)
    mix = D.Categorical(probs)
    comp = D.Normal(mus, stds)
    dist = D.MixtureSameFamily(mix, comp)
    return dist


def toy_cond_dist_bimodal(x):
    mu = 5.0 - 2 * x
    mus = torch.stack([mu, -mu], dim=-1)
    stds = torch.ones_like(mus)
    probs = torch.tensor([0.5, 0.5])[None, :].repeat(x.shape[0], 1)
    mix = D.Categorical(probs)
    comp = D.Normal(mus, stds)
    dist = D.MixtureSameFamily(mix, comp)
    return dist


def toy_cond_dist_cross(x):
    mu = 5.0 - 10.0 * x
    mus = torch.stack([mu, -mu], dim=-1)
    stds = torch.ones_like(mus)
    probs = torch.tensor([0.5, 0.5])[None, :].repeat(x.shape[0], 1)
    mix = D.Categorical(probs)
    comp = D.Normal(mus, stds)
    dist = D.MixtureSameFamily(mix, comp)
    return dist


def toy_cond_dist_sine_with_bars(x):
    mu1 = torch.sin(2 * torch.pi * x)
    mu2 = torch.full_like(mu1, 3.0)
    mu3 = torch.full_like(mu1, -3.0)
    mus = torch.stack([mu1, mu2, mu3], dim=-1)
    stds = torch.full_like(mus, 0.5)
    probs = torch.ones(3)[None, :].repeat(x.shape[0], 1)
    mix = D.Categorical(probs)
    comp = D.Normal(mus, stds)
    dist = D.MixtureSameFamily(mix, comp)
    return dist


def toy_cond_dist_discrete(x):
    n_classes = 6
    mus = torch.arange(n_classes, dtype=torch.float32)[None, :]
    mus = mus.repeat(x.shape[0], 1)
    stds = torch.full_like(mus, 0.001)
    probs = torch.arange(1, n_classes + 1, dtype=torch.float32)[None, :].repeat(x.shape[0], 1)
    mix = D.Categorical(probs)
    comp = D.Normal(mus, stds)
    dist = D.MixtureSameFamily(mix, comp)
    return dist


def get_toy_cond_dist(name):
    if name == 'simple':
        return toy_cond_dist_simple
    elif name == 'sine':
        return toy_cond_dist_sine
    elif name == 'rain':
        return toy_cond_dist_rain
    elif name == 'exp_varying_rate':
        return toy_cond_dist_exp_varying_rate
    elif name == 'trimodal':
        return toy_cond_dist_trimodal
    elif name == 'bimodal':
        return toy_cond_dist_bimodal
    elif name == 'cross':
        return toy_cond_dist_cross
    elif name == 'sine_with_bars':
        return toy_cond_dist_sine_with_bars
    elif name == 'discrete':
        return toy_cond_dist_discrete
    else:
        raise ValueError(f'Unknown toy distribution {name}')


class ToyDataModule(BaseDataModule):
    def use_known_uncertainty(self):
        return False

    def get_data(self):
        prefix, size = self.hparams.name.rsplit('_', 1)
        cond_dist = get_toy_cond_dist(prefix)
        x = torch.rand(int(size))
        y = cond_dist(x).sample()
        return x.numpy().reshape(-1, 1), y.numpy().reshape(-1, 1)
