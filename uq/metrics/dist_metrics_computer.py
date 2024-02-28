import torch

from uq.models.regul.marginal_regul import sample_spacing_entropy_estimation
from uq.utils.dist import icdf

from .calibration import (
    ece_on_worst_subgroups,
    get_observed_frequency,
    quantile_calibration_from_pits_with_sorting,
)
from .general import nll
from .independence import (
    delta_ils_coverage_from_models,
    indep_of_length_and_coverage_pearson,
)
from .metrics_computer import MetricsComputer
from .quantiles import length_and_coverage_from_quantiles, quantile_scores, wis


class DistMetricsComputer(MetricsComputer):
    def __init__(self, module, y=None, dist=None, **kwargs):
        super().__init__(module, **kwargs)
        self.y = y
        self.dist = dist

    def monitored_metrics(self):
        nll_value = nll(self.dist, self.y).mean()
        return {
            'nll': nll_value,
        }

    def cheap_metrics(self):
        nll_value = nll(self.dist, self.y).mean()
        pits = self.dist.cdf(self.y)
        calib_l1 = quantile_calibration_from_pits_with_sorting(pits, L=1)
        calib_l2 = quantile_calibration_from_pits_with_sorting(pits, L=2)
        calib_kl = -sample_spacing_entropy_estimation(pits, neural_sort=False)

        alpha = torch.arange(0.05, 1, 0.05)
        observed_frequency = get_observed_frequency(pits, alpha)
        observed_frequency_metrics = {
            f'observed_frequency_{level:.2f}': value for level, value in zip(alpha, observed_frequency)
        }

        return {
            'nll': nll_value,
            'calib_l1': calib_l1,
            'calib_l2': calib_l2,
            'calib_kl': calib_kl,
            **observed_frequency_metrics,
        }

    def costly_metrics(self):
        alpha = torch.arange(0.05, 1, 0.05)
        nan = torch.tensor(torch.nan)
        nan_vector = torch.full_like(self.y, torch.nan)

        quantile_scores_values = nan_vector
        quantile_scores_per_level = nan_vector
        pearson = nan
        wis_value = nan_vector
        quantiles_scores = {f'quantile_score_{level:.2f}': nan for level in alpha}
        length_90, coverage_90 = nan_vector, nan_vector
        median = nan_vector
        mean = nan_vector
        stddev = nan_vector

        compute_quantile_metrics = True
        try:
            quantiles = icdf(self.dist, alpha[:, None]).permute(1, 0)
        except AssertionError:
            # This happens when the distribution is ill-defined, only when both reflected and truncated are set to False.
            # In this case, we set the metrics based on quantile computation to nan.
            compute_quantile_metrics = False
        
        if compute_quantile_metrics:
            assert quantiles.shape == self.dist.batch_shape + alpha.shape
            quantile_scores_values = quantile_scores(self.y, quantiles, alpha)
            quantile_scores_per_level = quantile_scores_values.mean(dim=0)
            # We just take the mean for all coverage levels (same weight for each coverage level)
            pearson = indep_of_length_and_coverage_pearson(self.y, quantiles).mean()
            wis_value = wis(quantile_scores_values)
            quantiles_scores = {
                f'quantile_score_{level:.2f}': score for level, score in zip(alpha, quantile_scores_per_level)
            }
            length_90, coverage_90 = length_and_coverage_from_quantiles(self.y, quantiles, alpha, 0.05, 0.95)
            median_index = (alpha == 0.5).nonzero().item()
            median = quantiles[..., median_index]

        try:
            crps = self.dist.crps(self.y)
        except (NotImplementedError, AttributeError):
            crps = nan_vector
        try:
            mean = self.dist.mean
        except NotImplementedError:
            if compute_quantile_metrics:
                mean = quantiles.mean(dim=-1)
        try:
            stddev = self.dist.stddev
        except NotImplementedError:
            if compute_quantile_metrics:
                stddev = quantiles.std(dim=-1)

        return {
            'quantile_score': quantile_scores_per_level.mean(),
            'wis': wis_value.mean(),
            'crps': crps.mean(),
            'mean': mean.mean(),
            'mean_std': mean.std(),
            'stddev': stddev.mean(),
            'stddev_std': stddev.std(),
            'var': (stddev**2).mean(),
            'pearson': pearson,
            'length_90': length_90.mean(),
            'coverage_90': coverage_90.mean(),
            'rmse': (mean - self.y).square().mean().sqrt(),
            'mae': (median - self.y).abs().mean(),
            **quantiles_scores,
        }

    def train_metrics(self):
        metrics = self.cheap_metrics()
        if not self.rc.config.only_cheap_metrics:
            metrics = {**metrics, **self.costly_metrics()}
        return metrics

    def val_metrics(self):
        return self.train_metrics()

    def test_metrics(self):
        return self.val_metrics()


def test_conditional_metrics(dist_model, dist_baseline, y, prediction_fn, n_samples=1):
    worst_ece = ece_on_worst_subgroups(prediction_fn, y, n_samples=n_samples)
    (
        delta_ils_coverage_new,
        delta_ils_coverage_baseline,
    ) = delta_ils_coverage_from_models(dist_model, dist_baseline, y)
    return {
        'worst_ece': worst_ece.mean(),
        'delta_ils_coverage_new': delta_ils_coverage_new.mean(),
        'delta_ils_coverage_baseline': delta_ils_coverage_baseline.mean(),
    }
