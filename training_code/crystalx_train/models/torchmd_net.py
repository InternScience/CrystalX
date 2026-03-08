import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter

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
        position_noise_scale=0.,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise        
        self.position_noise_scale = position_noise_scale

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
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

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)


        # apply the output network
        x1 = self.output_model.pre_reduce(x, v, z, pos, batch)

        v = torch.norm(v, dim=-2)
        x = torch.cat([x, v], dim=-1)
        return x1, x
