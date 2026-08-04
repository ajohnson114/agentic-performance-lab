"""Correctness test: verify model produces valid output on a small batch."""
from __future__ import annotations

import torch
from dataset import SyntheticImageDataset
from model import SmallClassifier


def main():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    model = SmallClassifier(num_classes=10).to(device)
    model.eval()

    # Small dataset without simulated I/O delay for fast testing
    ds = SyntheticImageDataset(size=8, simulate_io_ms=0.0)
    images = torch.stack([ds[i][0] for i in range(8)]).to(device)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (8, 10), f"Unexpected shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Non-finite values in output"

    # Sanity: softmax sums to 1
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Softmax sums: {sums}"

    print("ok")


if __name__ == "__main__":
    main()
