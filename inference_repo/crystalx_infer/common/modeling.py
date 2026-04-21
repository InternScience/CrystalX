"""Model construction and checkpoint helpers for CrystalX inference."""

from __future__ import annotations

import torch

from crystalx_infer.common.runtime import unwrap_state_dict


DEFAULT_HIDDEN_CHANNELS = 256
DEFAULT_CHECKPOINT_HIDDEN_CHANNELS = 0


def infer_hidden_channels_from_state_dict(state_dict) -> int:
    """Infer the representation hidden width from a checkpoint."""

    embed_key = "representation_model.embedding.weight"
    if embed_key in state_dict and state_dict[embed_key].ndim == 2:
        return int(state_dict[embed_key].shape[1])

    norm_key = "representation_model.out_norm.weight"
    if norm_key in state_dict and state_dict[norm_key].ndim == 1:
        return int(state_dict[norm_key].shape[0])

    raise ValueError("Cannot infer hidden_channels from checkpoint.")


def infer_num_classes_from_state_dict(state_dict) -> int:
    """Infer the model output class count from a checkpoint."""

    bias_key = "output_model.output_network.1.update_net.2.bias"
    if bias_key in state_dict and state_dict[bias_key].ndim == 1:
        dim = int(state_dict[bias_key].shape[0])
        if dim % 2 == 0 and dim > 0:
            return dim // 2

    weight_key = "output_model.output_network.1.update_net.2.weight"
    if weight_key in state_dict and state_dict[weight_key].ndim == 2:
        dim = int(state_dict[weight_key].shape[0])
        if dim % 2 == 0 and dim > 0:
            return dim // 2

    vec_key = "output_model.output_network.1.vec2_proj.weight"
    if vec_key in state_dict and state_dict[vec_key].ndim == 2:
        return int(state_dict[vec_key].shape[0])

    raise ValueError("Cannot infer num_classes from checkpoint. Please pass --num_classes.")


def build_torchmd_model(
    num_classes: int,
    device,
    hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
):
    """Create a TorchMD-based CrystalX inference model."""

    from crystalx_infer.models.noise_output_model import EquivariantScalar
    from crystalx_infer.models.torchmd_et import TorchMD_ET
    from crystalx_infer.models.torchmd_net import TorchMD_Net

    representation_model = TorchMD_ET(
        hidden_channels=hidden_channels,
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
    )
    output_model = EquivariantScalar(hidden_channels, num_classes=num_classes)
    return TorchMD_Net(
        representation_model=representation_model,
        output_model=output_model,
    ).to(device)


def load_torchmd_model(
    model_path: str,
    device,
    num_classes: int | None = None,
    hidden_channels: int = DEFAULT_CHECKPOINT_HIDDEN_CHANNELS,
):
    """Load a TorchMD model checkpoint and return the ready-to-run model."""

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    inferred_hidden_channels = infer_hidden_channels_from_state_dict(state_dict)
    resolved_num_classes = (
        int(num_classes) if num_classes is not None and int(num_classes) > 0
        else infer_num_classes_from_state_dict(state_dict)
    )
    resolved_hidden_channels = (
        int(hidden_channels) if hidden_channels is not None and int(hidden_channels) > 0
        else inferred_hidden_channels
    )
    if resolved_hidden_channels != inferred_hidden_channels:
        raise ValueError(
            f"hidden_channels mismatch for checkpoint {model_path}: "
            f"requested={resolved_hidden_channels}, checkpoint={inferred_hidden_channels}."
        )
    model = build_torchmd_model(
        num_classes=resolved_num_classes,
        device=device,
        hidden_channels=resolved_hidden_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, resolved_num_classes
