"""Select an available accelerator before pinning an inference recipe."""

from __future__ import annotations


def inference_device() -> str:
    """Prefer CUDA, then Apple MPS, and otherwise use CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
