"""Correctness tests for the inference pipeline."""

import torch
from pipeline import SmallCNN, run_pipeline


def test_output_structure():
    """run_pipeline returns num_images dicts with 'class' (int 0-9) and 'confidence' (float 0-1)."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    results = run_pipeline(num_images=8, batch_size=1, device=device)

    assert len(results) == 8, f"Expected 8 results, got {len(results)}"
    for i, r in enumerate(results):
        assert "class" in r, f"Result {i} missing 'class' key"
        assert "confidence" in r, f"Result {i} missing 'confidence' key"
        assert isinstance(r["class"], int), f"Result {i} class is not int: {type(r['class'])}"
        assert 0 <= r["class"] <= 9, f"Result {i} class out of range: {r['class']}"
        assert isinstance(r["confidence"], float), f"Result {i} confidence is not float"
        assert 0.0 <= r["confidence"] <= 1.0, f"Result {i} confidence out of range: {r['confidence']}"


def test_model_output_shape():
    """SmallCNN produces (B, 10) output for various batch sizes."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SmallCNN().to(device)
    model.eval()

    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 3, 64, 64, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 10), f"Expected ({batch_size}, 10), got {out.shape}"


def test_determinism():
    """Two runs with the same seed produce identical class predictions."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(123)
    results_a = run_pipeline(num_images=8, batch_size=1, device=device)

    torch.manual_seed(123)
    results_b = run_pipeline(num_images=8, batch_size=1, device=device)

    classes_a = [r["class"] for r in results_a]
    classes_b = [r["class"] for r in results_b]
    assert classes_a == classes_b, f"Non-deterministic: {classes_a} != {classes_b}"


def test_confidence_is_probability():
    """Confidence values should be valid softmax outputs (positive, <= 1.0)."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SmallCNN().to(device)
    model.eval()

    x = torch.randn(4, 3, 64, 64, device=device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)

    # Each row should sum to ~1.0
    row_sums = probs.sum(dim=1)
    for i in range(4):
        assert abs(row_sums[i].item() - 1.0) < 1e-5, f"Row {i} sum = {row_sums[i].item()}"

    # All values should be in [0, 1]
    assert (probs >= 0).all(), "Negative probability found"
    assert (probs <= 1).all(), "Probability > 1 found"


def main():
    test_output_structure()
    print("ok  test_output_structure")

    test_model_output_shape()
    print("ok  test_model_output_shape")

    test_determinism()
    print("ok  test_determinism")

    test_confidence_is_probability()
    print("ok  test_confidence_is_probability")

    print("all tests passed")


if __name__ == "__main__":
    main()
