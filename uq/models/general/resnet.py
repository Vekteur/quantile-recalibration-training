# Code adapted from https://github.com/LeoGrin/tabular-benchmark/blob/main/src/models/tabular/bin/resnet.py

import typing as ty

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ResNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            d: int = 256,
            d_hidden_factor: float = 2,
            n_layers: int = 8,
            normalization: str = 'layernorm',
            hidden_dropout: float = 0.2,
            residual_dropout: float = 0.2,
            output_sizes: int = None,
            **kwargs,
    ) -> None:
        super().__init__()
        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)
        d_in = input_size
        self.main_activation = reglu
        self.last_activation = F.relu
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout
        d_out = output_sizes
        assert d_out is not None
        self.d_out = d_out

        d_hidden = int(d * d_hidden_factor)

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * 2 # We multiply by 2 because of reglu activation
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, sum(d_out))

    def forward(self, x, r=None) -> Tensor:
        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = torch.split(x, self.d_out, dim=-1)
        return x

