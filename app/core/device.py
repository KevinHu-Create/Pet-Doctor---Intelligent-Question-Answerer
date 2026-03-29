from functools import lru_cache

import torch

from app.core.settings import settings


@lru_cache
def get_torch_device() -> str:
    if settings.TORCH_DEVICE != "auto":
        return settings.TORCH_DEVICE

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"
