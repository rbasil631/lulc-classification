"""UNet++ / ResNet-50 segmentation model.

The encoder is initialised from TorchGeo's SSL4EO Sentinel-2 MoCo weights when
available. Those were trained on millions of Sentinel-2 scenes, so they carry
far more useful priors for 6-band multispectral input than ImageNet RGB
weights; ImageNet is only the fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from lulc.utils import get_device


def _load_torchgeo_weights(model: nn.Module, verbose: bool = True) -> bool:
    """Copy SSL4EO weights into the encoder for every shape-compatible layer."""
    try:
        from torchgeo.models import ResNet50_Weights
        from torchgeo.models import resnet50 as tg_resnet50

        tg_state = tg_resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO).state_dict()
        enc_state = model.encoder.state_dict()
        matched = {
            k: v
            for k, v in tg_state.items()
            if k in enc_state and v.shape == enc_state[k].shape
        }
        enc_state.update(matched)
        model.encoder.load_state_dict(enc_state, strict=False)
        if verbose:
            print(f"  Encoder: SSL4EO weights, {len(matched)}/{len(enc_state)} tensors matched")
        return True
    except Exception as exc:  # torchgeo missing, or no network for the download
        if verbose:
            print(f"  Encoder: SSL4EO unavailable ({exc}); using ImageNet init")
        return False


def build_model(
    num_bands: int = 6,
    num_classes: int = 6,
    encoder_weights: str | None = "imagenet",
    use_torchgeo: bool = True,
    device: torch.device | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Construct UNet++ with a ResNet-50 encoder.

    Parameters
    ----------
    encoder_weights : str | None
        Passed to ``smp``; set to ``None`` when weights will come from a
        checkpoint anyway (inference), which avoids a pointless download.
    use_torchgeo : bool
        Attempt to overwrite the encoder with SSL4EO Sentinel-2 weights.
    """
    device = device or get_device()

    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=encoder_weights,
        in_channels=num_bands,
        classes=num_classes,
        activation=None,
    )

    if use_torchgeo:
        _load_torchgeo_weights(model, verbose=verbose)

    model.to(device)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters: {n_params:.1f}M | Device: {device}")
    return model


def load_checkpoint(
    path: str | os.PathLike,
    num_bands: int = 6,
    num_classes: int = 6,
    device: torch.device | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Rebuild the architecture and load a saved ``state_dict``.

    Uses ``weights_only=True``: a checkpoint is untrusted input, and the
    default unpickling path can execute arbitrary code.
    """
    device = device or get_device()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = build_model(
        num_bands=num_bands,
        num_classes=num_classes,
        encoder_weights=None,
        use_torchgeo=False,
        device=device,
        verbose=False,
    )
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Checkpoint: {path.name}")
        print(f"  Parameters: {n_params:.1f}M | Device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    return model
