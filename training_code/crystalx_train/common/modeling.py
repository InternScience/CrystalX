from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationConfig:
    hidden_channels: int = 512
    attn_activation: str = "silu"
    num_heads: int = 8
    distance_influence: str = "both"


def build_model(rep_config: RepresentationConfig, num_classes: int) -> TorchMD_Net:
    from crystalx_train.models.noise_output_model import EquivariantScalar
    from crystalx_train.models.torchmd_et import TorchMD_ET
    from crystalx_train.models.torchmd_net import TorchMD_Net

    representation_model = TorchMD_ET(
        hidden_channels=rep_config.hidden_channels,
        attn_activation=rep_config.attn_activation,
        num_heads=rep_config.num_heads,
        distance_influence=rep_config.distance_influence,
    )
    output_model = EquivariantScalar(
        rep_config.hidden_channels,
        num_classes=num_classes,
    )
    return TorchMD_Net(
        representation_model=representation_model,
        output_model=output_model,
    )
