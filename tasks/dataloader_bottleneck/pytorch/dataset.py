"""Synthetic dataset with configurable CPU-heavy transform.

Generates random image-like tensors (3x224x224). The transform includes
a deliberate time.sleep to simulate slow I/O / CPU preprocessing, making
the DataLoader the bottleneck when num_workers=0.
"""
from __future__ import annotations

import time

import torch
from torch.utils.data import Dataset


class SyntheticImageDataset(Dataset):
    """Produces random 3x224x224 tensors with a CPU-heavy transform."""

    def __init__(self, size: int = 4096, simulate_io_ms: float = 1.0):
        self.size = size
        self.simulate_io_ms = simulate_io_ms
        # Pre-generate random seeds per sample for reproducibility
        self._seeds = torch.randint(0, 2**31, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # Simulate slow I/O or CPU-heavy preprocessing
        if self.simulate_io_ms > 0:
            time.sleep(self.simulate_io_ms / 1000.0)

        # Deterministic random tensor from seed
        gen = torch.Generator()
        gen.manual_seed(int(self._seeds[idx]))
        image = torch.randn(3, 224, 224, generator=gen)

        # Simple "normalize" transform (CPU work)
        image = (image - image.mean()) / (image.std() + 1e-7)

        # Random crop to 200x200 (CPU work)
        top = torch.randint(0, 24, (1,), generator=gen).item()
        left = torch.randint(0, 24, (1,), generator=gen).item()
        image = image[:, top:top + 200, left:left + 200]

        # Simple label (classification target)
        label = idx % 10
        return image, label
