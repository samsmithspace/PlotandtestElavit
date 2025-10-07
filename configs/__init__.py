# configs/__init__.py
from accelerate.commands.config.config_args import BaseConfig

from .default import DefaultConfig
from .model_configs import ViTBaseConfig, ViTLargeConfig, ViTSmallConfig,Food101Config,BeansConfig,CIFAR100Config,DeiTBeansConfig,DeiTConfig,DeiTCIFAR100Config,DeiTFoodConfig,DefaultConfig

__all__ = [
    'DefaultConfig',
    'ViTBaseConfig',
    'ViTLargeConfig',
    'ViTSmallConfig',
    'Food101Config',
    'BaseConfig',
    'CIFAR100Config',
    'DeiTBeansConfig',
    'DeiTConfig',
    'DeiTFoodConfig'
]
