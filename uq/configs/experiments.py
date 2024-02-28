from omegaconf import OmegaConf

from uq import utils

log = utils.get_logger(__name__)


def uci(config):
    config.dataset_groups = dict(uci=config.dataset_groups.uci)


def configure_experiments(config):
    if config.name:
        if config.name == 'uci':
            uci(config)
