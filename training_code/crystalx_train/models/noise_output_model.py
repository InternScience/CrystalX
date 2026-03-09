from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from crystalx_train.models.torchmd_utils import GatedEquivariantBlock


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super().__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        raise NotImplementedError

    def post_reduce(self, x):
        return x


class EquivariantScalar(OutputModel):
    def __init__(self, hidden_channels, num_classes=1, activation="silu", allow_prior_model=True):
        super().__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    num_classes,
                    activation=activation,
                ),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return x + v.sum() * 0

    def vis_mid(self, x, v, z, pos, batch):
        layer1 = self.output_network[0]
        layer2 = self.output_network[1]
        x1, v = layer1(x, v)
        x2, v = layer2(x1, v)
        return x1, x2
