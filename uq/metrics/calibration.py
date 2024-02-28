import numpy as np
import torch

from uq.models.regul.marginal_regul import (
    cdf_based_regul_from_pit,
    cdf_based_regul_from_quantiles,
    quantile_based_regul,
)
from uq.utils.torch_utils import centered_bins


def get_observed_frequency(pit, alpha):
    indicator = pit[:, None] <= alpha
    return indicator.float().mean(dim=0)


def quantile_calibration_from_pits(pit, L, alpha):
    return cdf_based_regul_from_pit(pit, L, alpha, s=None)


def quantile_calibration_from_pits_with_sorting(pit, L):
    return quantile_based_regul(pit, L, neural_sort=False)


def quantile_calibration_from_quantiles(quantiles, y, L, alpha):
    return cdf_based_regul_from_quantiles(quantiles, y, L, alpha)


def ece(pit):
    # Similar to quantile_calibration_from_pits_with_sorting(pit, L=1) without sorting
    lin = centered_bins(pit.shape[0])
    return (pit - lin).abs().mean()


def ece_on_worst_subgroups(random_dist, y, n_samples=1, size_ratio=0.05):
    """
    We take the PIT (after (group) post-hoc calibration) of each point multiple times.
    We find which points, in average over the model, give the smallest and highest PIT.
    We select non-averaged PIT of these points and evaluate their ECE.
    """
    pit_list = []
    for _ in range(n_samples):
        pit = random_dist().cdf(y)
        pit_list.append(pit)

    # We want the mean pit of each point
    pit_avg = torch.stack(pit_list, dim=1).mean(dim=1)
    _, sorted_indices = torch.sort(pit_avg)
    pit = pit_list[0]   # Just take any pit from the list

    N = y.shape[0]
    size = int(N * size_ratio)
    pit_smallest = pit[sorted_indices[:size]]
    pit_highest = pit[sorted_indices[-size:]]

    return (ece(pit_smallest) + ece(pit_highest)) / 2


def group_posthoc_calibration():
    """
    We fix group_idx, which is a dimension of x.
    We divide the data into two groups: the points that are lower and higher than the median on the dimension group_idx.
    We recalibrate each group separately.
    """


def eval_ece_by_2dim(x, y, model):
    """
    We consider each pair of dimensions of x.
    For each pair, we divide the data into 4 groups (according to the relation with the median of both dimensions).
    We take the PIT of each point (only one time in this case, why?).
    The error is recomputed to avoid overfitting (but I think that max_err should be updated earlier to really avoid overfitting).
    We return the pair of dimensions and the group that gives the highest error.
    """

    size = int(x.shape[0])
    size_lb = size // 4
    size_ub = size - size_lb

    num_feat = int(x.shape[1])
    mid_point = [torch.sort(x[:, i])[0][size // 2] for i in range(num_feat)]

    max_err = -1
    worst_pit = None
    worst_dim = None
    for i in range(num_feat):
        for j in range(i + 1, num_feat):
            mask_i = x[:, i] < mid_point[i]
            mask_j = x[:, j] < mid_point[j]
            index1 = mask_i & mask_j
            index2 = mask_i & ~mask_j
            index3 = ~mask_i & mask_j
            index4 = ~mask_i & ~mask_j

            for k, index in enumerate([index1, index2, index3, index4]):
                test_x_part, test_y_part = x[index], y[index]
                if test_x_part.shape[0] < size_lb or test_x_part.shape[0] > size_ub:
                    continue

                with torch.no_grad():
                    pit = model.eval_all(test_x_part, test_y_part)[0].cpu().numpy()[:, 0]
                    model.apply_recalibrate(pit, test_x_part)
                    pit = np.sort(pit)

                    # Compute calibration error
                    err = np.mean(np.abs(pit - np.linspace(0, 1, pit.shape[0])))

                    if err > max_err:
                        # Re-evaluate to prevent over-fitting
                        pit = model.eval_all(test_x_part, test_y_part)[0].cpu().numpy()[:, 0]
                        model.apply_recalibrate(pit, test_x_part)
                        pit = np.sort(pit)

                        # Compute calibration error
                        err = np.mean(np.abs(pit - np.linspace(0, 1, pit.shape[0])))

                        max_err = err
                        worst_pit = pit
                        worst_dim = [i, j, k]

    return max_err, worst_dim
