from copy import copy

import torch

from uq.models.posthoc.posthoc_model import get_posthoc_transformer

from .metrics_collector import MetricsCollector


class PosthocModule:
    def __init__(self, base_module, posthoc_dataset, hparams):
        self.base_module = base_module
        self.hparams = hparams
        self.collector = MetricsCollector(base_module)
        self.model = get_posthoc_transformer(base_module, base_module.module, posthoc_dataset, hparams)

    def build(self, epoch, batch_idx, stage, batch=None):
        self.model.build(epoch, batch_idx, stage, batch=batch)

    def make_run_config(self):
        rc = copy(self.base_module.rc)
        hparams_with_prefix = {f'posthoc_{key}': value for key, value in self.hparams.items()}
        rc.hparams = {**rc.hparams, **hparams_with_prefix}
        # Convert any defaultdict to dict due to a bug (https://github.com/python/cpython/issues/79721)
        rc.metrics = dict(self.collector.metrics)
        return rc


class PostHocModuleManager:
    def __init__(self, base_module, posthoc_dataset, posthoc_grid):
        self.base_module = base_module
        self.modules = [PosthocModule(base_module, posthoc_dataset, hparams) for hparams in posthoc_grid]

    def get_module(self, hparams):
        for module in self.modules:
            if module.hparams == hparams:
                return module
        raise ValueError(f'No module with hparams {hparams} found.')

    def collect_per_step(self, outputs, stage):
        outputs_per_module = [[] for _ in self.modules]
        for outputs_at_step in outputs:
            for i, outputs_for_some_module in enumerate(outputs_at_step['posthoc_metrics_list']):
                outputs_per_module[i].append(outputs_for_some_module)

        for module, outputs in zip(self.modules, outputs_per_module):
            module.collector.collect_per_step(outputs, stage)

    def add_best_iter_metrics(self):
        for module in self.modules:
            module.collector.add_best_iter_metrics()

    def advance_timer(self, timer, amount, cancel_time=False):
        for module in self.modules:
            module.collector.advance_timer(timer, amount, cancel_time=cancel_time)

    def build(self, epoch, batch_idx):
        for module in self.modules:
            module.build(epoch, batch_idx)

    def make_run_configs(self):
        return [module.make_run_config() for module in self.modules]
