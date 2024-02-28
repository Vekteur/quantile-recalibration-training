from collections import defaultdict

import torch


class MetricsCollector:
    def __init__(self, module):
        self.metrics = defaultdict(lambda: defaultdict(dict))
        self.module = module

    def collect_per_step(self, outputs, stage):
        """
        Collect metrics as returned by PyTorch Lightning.
        `outputs` is a `list` (containing one element per step) of `dict`
        (containing all metrics at this step).
        """
        metrics_at_all_steps = defaultdict(list)
        for metrics_at_some_step in outputs:
            for key, value in metrics_at_some_step.items():
                metrics_at_all_steps[key].append(value)
        for key, values in metrics_at_all_steps.items():
            mean_value = torch.stack(values, dim=0).mean(dim=0).item()
            self.metrics['per_epoch'][f'{stage}_{key}'][self.module.current_epoch] = mean_value

    def get_best_score_and_iter(self):
        best_score = float('inf')
        best_iter = -1
        for iter, score in self.metrics['per_epoch'][f'val_{self.module.monitor}'].items():
            if score < best_score:
                best_score = score
                best_iter = iter
        return best_score, best_iter

    def add_best_iter_metrics(self):
        (
            self.metrics['best_score'],
            self.metrics['best_iter'],
        ) = self.get_best_score_and_iter()

    def advance_timer(self, timer, amount, cancel_time=False):
        if timer not in self.metrics:
            self.metrics[timer] = 0
        if cancel_time:
            self.metrics[timer] -= amount
        else:
            self.metrics[timer] += amount
        #print('new value of', timer, 'is', self.metrics[timer])
