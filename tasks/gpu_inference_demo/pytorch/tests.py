"""Correctness tests for the GPU inference demo pipeline."""

import torch
from pipeline import run_pipeline


def test_output_structure():
    """run_pipeline returns dicts with 'class_id' (int 0-999) and 'confidence' (float 0-1)."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    results = run_pipeline(num_images=8, batch_size=4, device=device)

    assert len(results) == 8, f"Expected 8 results, got {len(results)}"
    for i, r in enumerate(results):
        assert "class_id" in r, f"Result {i} missing 'class_id' key"
        assert "confidence" in r, f"Result {i} missing 'confidence' key"
        assert isinstance(r["class_id"], int), (
            f"Result {i} class_id is not int: {type(r['class_id'])}"
        )
        assert 0 <= r["class_id"] <= 999, (
            f"Result {i} class_id out of range: {r['class_id']}"
        )
        assert isinstance(r["confidence"], float), (
            f"Result {i} confidence is not float: {type(r['confidence'])}"
        )
        assert 0.0 <= r["confidence"] <= 1.0, (
            f"Result {i} confidence out of range: {r['confidence']}"
        )


def test_output_count():
    """Verify correct number of results for different num_images values."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    for n in [1, 7, 16, 33]:
        results = run_pipeline(num_images=n, batch_size=8, device=device)
        assert len(results) == n, (
            f"Expected {n} results, got {len(results)}"
        )


def test_determinism():
    """Two runs with the same seed produce identical class predictions."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    torch.manual_seed(42)
    results_a = run_pipeline(num_images=16, batch_size=4, device=device)

    torch.manual_seed(42)
    results_b = run_pipeline(num_images=16, batch_size=4, device=device)

    classes_a = [r["class_id"] for r in results_a]
    classes_b = [r["class_id"] for r in results_b]
    assert classes_a == classes_b, f"Non-deterministic: {classes_a} != {classes_b}"


def test_confidence_valid():
    """Confidence values are valid probabilities (0-1, softmax row sums ≈ 1)."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    results = run_pipeline(num_images=8, batch_size=4, device=device)

    for i, r in enumerate(results):
        c = r["confidence"]
        assert 0.0 <= c <= 1.0, (
            f"Result {i} confidence {c} not in [0, 1]"
        )
        # Max confidence from softmax over 1000 classes should be > 0
        assert c > 0.0, f"Result {i} confidence is exactly 0"


def main():
    test_output_structure()
    print("ok  test_output_structure")

    test_output_count()
    print("ok  test_output_count")

    test_determinism()
    print("ok  test_determinism")

    test_confidence_valid()
    print("ok  test_confidence_valid")

    print("\nall tests passed")


if __name__ == "__main__":
    main()
