"""Inference pipeline: SmallCNN model with naive preprocessing and postprocessing.

Processes synthetic 64x64 RGB images through a small CNN classifier.
The baseline is deliberately slow: per-image CPU preprocessing, batch_size=1,
eager mode, fp32, and per-image .cpu() synchronization in postprocessing.
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Small 10-class CNN classifier (~267K parameters).

    Input: (B, 3, 64, 64) float32
    Output: (B, 10) logits
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def preprocess_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess a single image: normalize to [0,1], apply channel-wise standardization."""
    img = image_tensor.clone().float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    return img


def postprocess_output(logits: torch.Tensor) -> list[dict]:
    """Convert logits to predictions — per-image .cpu() and softmax."""
    results = []
    for i in range(logits.shape[0]):
        probs = torch.softmax(logits[i].cpu(), dim=0)
        top_val, top_idx = probs.topk(1)
        results.append({"class": top_idx.item(), "confidence": top_val.item()})
    return results


def run_pipeline(num_images: int, batch_size: int, device: str) -> list[dict]:
    """Run inference pipeline over synthetic images.

    Returns a list of num_images dicts, each with 'class' (int) and 'confidence' (float).
    """
    model = SmallCNN().to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

    all_results: list[dict] = []
    with torch.inference_mode(), torch.amp.autocast(device_type="mps", dtype=torch.float16):
        for start in range(0, num_images, batch_size):
            bsz = min(batch_size, num_images - start)

            # Generate synthetic images directly on device (float16 in [0,1]).
            # Equivalent to uint8 randint then /255 (distribution is slightly different but still synthetic).
            x = torch.rand((bsz, 3, 64, 64), device=device, dtype=torch.float16)
            x.sub_(mean).div_(std)

            logits = model(x)

            probs = torch.softmax(logits, dim=1)
            top_val, top_idx = probs.max(dim=1)
            top_val = top_val.detach().cpu()
            top_idx = top_idx.detach().cpu()
            all_results.extend(
                [{"class": int(top_idx[i]), "confidence": float(top_val[i])} for i in range(top_idx.numel())]
            )
    return all_results
