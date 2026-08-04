"""Inference pipeline: ResNet-50 with deliberately slow preprocessing and postprocessing.

Processes synthetic 224x224 RGB images through a ResNet-50 classifier (1000 classes).
The baseline is deliberately unoptimized with 14 antipatterns the agent
should identify and fix for a 5-15x speedup on H100.

Antipatterns:
  1. CPU tensor creation + per-iteration .to(device)
  2. Per-image preprocessing loop
  3. FP32 only — no autocast / half precision
  4. torch.no_grad() instead of torch.inference_mode()
  5. .item() in hot path — forces CUDA sync
  6. .cpu() before argmax — unnecessary device transfer
  7. model.eval() called every invocation — redundant
  8. No torch.compile() — pure eager mode
  9. Redundant .clone().contiguous() on already-contiguous tensors
  10. Sequential postprocessing — Python loop instead of vectorized ops
  11. No channels_last memory format on input tensors
  12. No pinned memory for H2D transfer
  13. Model not converted to channels_last memory format
  14. No CUDA graphs for repeated fixed-shape inference
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50


def load_model(device: str) -> nn.Module:
    """Load ResNet-50 (random weights, 1000 classes).

    Always creates a fresh model instance — no caching.
    """
    # Antipattern 13: model is not converted to channels_last memory format.
    # model.to(memory_format=torch.channels_last) enables NHWC layout for conv ops,
    # which is significantly faster on H100 tensor cores.
    model = resnet50(weights=None, num_classes=1000)
    model = model.to(device)
    # Antipattern 7: model.eval() is called here but also redundantly
    # called every time in run_pipeline
    model.eval()
    return model


def preprocess_batch(images: torch.Tensor, device: str) -> torch.Tensor:
    """Normalize and transfer a batch of images to device.

    Args:
        images: (N, 3, 224, 224) uint8 tensor on CPU
        device: target device string

    Returns:
        Normalized float32 tensor on device.
    """
    # Antipattern 9: redundant clone and contiguous on fresh tensors
    images = images.clone().contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    processed = []
    # Antipattern 2: per-image preprocessing loop instead of batched
    for i in range(images.shape[0]):
        img = images[i].float() / 255.0
        for c in range(3):
            img[c] = (img[c] - mean[c]) / std[c]
        processed.append(img)

    batch = torch.stack(processed)
    # Antipattern 1: transfer to device after CPU work
    # Antipattern 11: batch is in default NCHW (contiguous) memory format.
    # Converting to channels_last via batch.to(memory_format=torch.channels_last)
    # enables faster conv kernels on H100 tensor cores.
    # Antipattern 12: using pageable CPU memory for H2D transfer.
    # pin_memory() + non_blocking=True would overlap transfer with compute.
    batch = batch.to(device)
    return batch


def postprocess_outputs(logits: torch.Tensor) -> list[dict]:
    """Extract predictions from logits.

    Args:
        logits: (N, 1000) tensor on device

    Returns:
        List of dicts with 'class_id' (int 0-999) and 'confidence' (float 0-1).
    """
    results = []
    # Antipattern 10: sequential postprocessing in Python loop
    for i in range(logits.shape[0]):
        # Antipattern 6: .cpu() before argmax — unnecessary device transfer
        single = logits[i].cpu()
        probs = torch.softmax(single, dim=0)
        # Antipattern 5: .item() forces CUDA sync (already on cpu here,
        # but the .cpu() above was the real sync point)
        max_prob = probs.max().item()
        max_idx = probs.argmax().item()
        results.append({"class_id": int(max_idx), "confidence": float(max_prob)})
    return results


def run_pipeline(
    num_images: int, batch_size: int, device: str
) -> list[dict]:
    """Run full inference pipeline over synthetic images.

    Args:
        num_images: total number of images to process
        batch_size: images per forward pass
        device: 'cuda', 'cpu', or 'mps'

    Returns:
        List of num_images dicts, each with 'class_id' (int 0-999)
        and 'confidence' (float 0-1).
    """
    model = load_model(device)
    # Antipattern 7: redundant model.eval() — already called in load_model
    model.eval()

    all_results: list[dict] = []

    # Antipattern 4: torch.no_grad() instead of torch.inference_mode()
    # Antipattern 3: no autocast — everything runs in FP32
    # Antipattern 8: no torch.compile() — pure eager mode
    # Antipattern 14: no CUDA graphs — for fixed batch sizes, CUDA graphs
    # eliminate kernel launch overhead entirely. The agent should discover
    # graph capture after fixing other antipatterns.
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            bsz = min(batch_size, num_images - start)

            # Antipattern 1: create synthetic images on CPU, then transfer
            images = torch.randint(0, 256, (bsz, 3, 224, 224), dtype=torch.uint8)

            batch = preprocess_batch(images, device)
            logits = model(batch)
            results = postprocess_outputs(logits)
            all_results.extend(results)

    return all_results
