import torch
from torch import nn

from uq.utils.general import elapsed_timer

from ..decorator_module import ComponentModule
from ..general.mlp import MixturePrediction, SplinePrediction


class NeuralNetworkModule(ComponentModule):
    def __init__(self, base_module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_module = [base_module]

    def predict(self, x):
        with elapsed_timer() as time:
            pred = self.model.dist(x)
        self.base_module.advance_timer('nn_time', time())
        return pred

    def step(self, x, y, batch_idx, stage):
        return self.predict(x)

    def build_model(self):
        pass

    def get_mlp_args(self):
        return dict(
            input_size=self.input_size,
            hidden_sizes=[self.units_size for _ in range(self.nb_hidden)],
            drop_prob=self.drop_prob,
            base_model=self.base_model,
        )

    def build(self):
        self.model = self.build_model()

    def capture_hparams(
        self, input_size=None, units_size=128, nb_hidden=3, drop_prob=0.2, misspecification=None, base_model='nn', **kwargs
    ):
        self.input_size = input_size
        assert input_size is not None
        self.units_size = units_size
        self.nb_hidden = nb_hidden
        self.drop_prob = drop_prob
        self.misspecification = misspecification
        self.base_model = base_model
        self.randomized_predictions = False

        if self.misspecification == 'small_mlp':
            self.nb_hidden = 1
            self.units_size = 32
        elif self.misspecification == 'big_mlp':
            self.nb_hidden = 30


class MixtureModule(NeuralNetworkModule):
    def build_model(self):
        homoscedastic = self.misspecification == 'homoscedasticity'
        return MixturePrediction(
            mixture_size=self.mixture_size, homoscedastic=homoscedastic, **self.get_mlp_args()
        )

    def capture_hparams(self, mixture_size=3, **kwargs):
        super().capture_hparams(**kwargs)
        self.mixture_size = mixture_size


class SplineModule(NeuralNetworkModule):
    def build_model(self):
        assert not self.misspecification == 'homoscedasticity'
        return SplinePrediction(count_bins=self.count_bins, **self.get_mlp_args())

    def capture_hparams(self, count_bins=6, **kwargs):
        super().capture_hparams(**kwargs)
        self.count_bins = count_bins
