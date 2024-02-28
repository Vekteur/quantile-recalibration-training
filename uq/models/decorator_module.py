from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import Tensor, nn
from torch.nn.modules.module import Module


class ComponentModule(nn.Module, ABC):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.loss = 0
        self.metrics = {}

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def step(self, x, y, batch_idx, stage):
        pass

    def build(self):
        pass

    def propagate_hparams(self, hparams):
        self.capture_hparams(**hparams)

    def capture_hparams(self, **kwargs):
        pass

    def before_train_starts(self):
        pass

    def get_loss(self):
        loss = self.loss
        self.loss = 0
        return loss

    def get_metrics(self):
        metrics = self.metrics
        self.metrics = {}
        return metrics

    def get_name(self, name):
        return self if self.name == name else None

    # Workaround to have a reference to the base module without registering it as a submodule
    # and causing a recursion error
    @property
    def base_module(self):
        return self._base_module[0]


class DecoratorModule(ComponentModule):
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
        self._base_module = self.module._base_module

    def build(self):
        super().build()
        self.module.build()

    def predict(self, x):
        return self.module.predict(x)

    def step(self, x, y, batch_idx, stage):
        return self.module.step(self, x, y, batch_idx, stage)

    def propagate_hparams(self, hparams):
        super().propagate_hparams(hparams)
        self.module.propagate_hparams(hparams)

    def before_train_starts(self):
        super().before_train_starts()
        self.module.before_train_starts()

    def get_loss(self):
        return super().get_loss() + self.module.get_loss()

    def get_metrics(self):
        return {**self.module.get_metrics(), **super().get_metrics()}

    def get_name(self, name):
        return self if self.name == name else self.module.get_name(name)
