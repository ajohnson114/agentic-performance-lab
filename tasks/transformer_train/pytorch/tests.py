"""Correctness test: small transformer, 5 training steps, loss finite and decreasing."""

import torch
from model import SmallTransformer


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(42)
    model = SmallTransformer(
        vocab_size=1024,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for step in range(10):
        tokens = torch.randint(0, 1024, (4, 32), device=device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 1024), targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Check: all losses finite
    for i, l in enumerate(losses):
        assert not (l != l), f"Loss at step {i} is NaN"  # NaN != NaN
        assert l < float("inf"), f"Loss at step {i} is infinite"

    # Check: loss is decreasing overall (last < first)
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )

    # Logit sanity check: verify model architecture produces reasonable outputs
    with torch.no_grad():
        test_tokens = torch.randint(0, 1024, (1, 32), device=device)
        logits = model(test_tokens[:, :-1])
        logit_mean = logits.float().mean().item()
        logit_std = logits.float().std().item()
        assert -5 <= logit_mean <= 5, f"Logit mean out of range: {logit_mean}"
        assert 0.01 <= logit_std <= 50, f"Logit std out of range: {logit_std}"

    print(f"ok  losses={[f'{l:.4f}' for l in losses]}")


if __name__ == "__main__":
    main()
