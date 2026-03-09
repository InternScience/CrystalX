from typing import Optional

import torch
from torch import nn


class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.0,
    ):
        super().__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.prior_model = prior_model
        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise
        self.position_noise_scale = position_noise_scale

        mean = torch.scalar_tensor(0) if mean is None else mean
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, z, pos, batch: Optional[torch.Tensor] = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)
        x_out = self.output_model.pre_reduce(x, v, z, pos, batch)
        v_norm = torch.norm(v, dim=-2)
        hidden = torch.cat([x, v_norm], dim=-1)
        return x_out, hidden
