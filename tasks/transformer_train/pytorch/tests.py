"""Correctness + anti-gaming test: small transformer, 10 training steps.

Beyond "loss goes down", this file wires in the perflab.harness reward-hack
mitigations so a candidate cannot win the benchmark by faking the work:

  * assert_real_tensor   — the model must return a genuine, materialized
                           torch.Tensor, not a lazy subclass that defers the
                           real computation until a comparison operator runs.
  * assert_deterministic — identical inputs must give identical outputs
                           (catches stale/uninitialized output buffers), and
                           different inputs must give different outputs
                           (catches no-op kernels).
  * in-place poison      — mutating the input tensor's storage without
                           changing its address must change the output
                           (catches a cache keyed on tensor data_ptr).
  * master-weight dtype  — parameters must stay fp32.

PRECISION POLICY (deliberate — read before tightening):
AMP / bf16 / fp16 autocast is the *intended* winning optimization for this
task (task.yaml declares data_hints.dtype_safety: bf16_safe). So this file
must NOT use assert_ulp_close against an fp64 reference on the model output:
under autocast the logits are legitimately bf16/fp16, many ULP away from an
fp64 reference, and such a check would reject exactly the solution the task
is designed to reward.

What we check instead is what mixed precision must NOT break:
  1. Parameters stay fp32. Real AMP keeps fp32 master weights and autocasts
     only the ops; a blanket model.half() throws away gradient precision and
     is an accuracy regression, not an optimization. This check passes for
     autocast and fails for .half().
  2. Reproducibility is checked at the harness default strict tolerance, NOT a
     loosened mixed-precision one. It reruns the same binary on the same
     inputs, so a correct implementation is bit-identical whatever its compute
     dtype; loosening it would let uninitialized-buffer garbage slip through.
  3. Things that genuinely are accuracy questions (loss decreasing, logit
     scale) use bounds wide enough for bf16 rounding.
"""

import torch
from model import SmallTransformer

from perflab.harness import assert_deterministic, assert_real_tensor

VOCAB_SIZE = 1024
SEQ_LEN = 32
BATCH = 4


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(42)
    model = SmallTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=SEQ_LEN,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _step in range(10):
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = model(inputs)
        # Anti-gaming: the forward output must be a real, materialized tensor.
        # A torch.Tensor subclass that captures its inputs and only computes
        # when compared would make the benchmark look arbitrarily fast.
        assert_real_tensor(logits, name="model logits")
        # .float() so the loss is accumulated in fp32 even when the model
        # returns autocast bf16/fp16 logits — that is what AMP itself does.
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE).float(), targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Check: all losses finite
    for i, loss_i in enumerate(losses):
        assert not (loss_i != loss_i), f"Loss at step {i} is NaN"  # NaN != NaN
        assert loss_i < float("inf"), f"Loss at step {i} is infinite"

    # Check: loss is decreasing overall (last < first)
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )

    # --- Anti-gaming: master weights must stay fp32 (see PRECISION POLICY) ---
    for pname, param in model.named_parameters():
        assert param.dtype == torch.float32, (
            f"Parameter '{pname}' has dtype {param.dtype}, expected torch.float32. "
            f"Use autocast/AMP — which keeps fp32 master weights and casts only "
            f"the ops — rather than casting the model itself to a low-precision "
            f"dtype, which destroys gradient precision."
        )

    # --- Anti-gaming: determinism and no-op detection ---
    # eval() + no_grad so this measures the forward computation only and does
    # not accumulate an autograd graph across the repeated runs.
    model.eval()
    with torch.no_grad():
        # Tolerance deliberately left at the harness default (strict): this
        # reruns the same binary on the same inputs, so even a bf16 autocast
        # model is bit-identical run to run. See PRECISION POLICY.
        assert_deterministic(
            fn=lambda toks: model(toks),
            input_factory=lambda: (
                torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN - 1), device=device),
            ),
            n_runs=3,
        )

        # In-place poison: overwrite the SAME storage (data_ptr unchanged) with
        # new tokens. A cache keyed on the input tensor's address would return
        # the previous logits; a real forward pass cannot.
        #
        # This is the pointer_poison mitigation adapted to a whole-model task.
        # perflab.harness.assert_no_memoization is not called directly because
        # it requires an independent reference_fn, and any reference for this
        # model would have to live in model.py — the very file the agent may
        # rewrite, so it would be gamed in lockstep. Comparing against the
        # pre-poison output needs no reference and is immune to that.
        poison_tokens = torch.randint(
            0, VOCAB_SIZE, (BATCH, SEQ_LEN - 1), device=device
        )
        before = model(poison_tokens).detach().clone().float()
        poison_tokens.data.copy_(
            torch.randint_like(poison_tokens, low=0, high=VOCAB_SIZE)
        )
        after = model(poison_tokens).detach().clone().float()
        assert not torch.equal(before, after), (
            "Memoization detected: the model returned identical logits after the "
            "input tokens were overwritten in place. This indicates a cache keyed "
            "on the input tensor's data pointer rather than a real forward pass."
        )

        # Logit sanity check. Bounds are wide enough for bf16/fp16 autocast
        # rounding — this is an accuracy question, so unlike the reproducibility
        # check above it is deliberately tolerant.
        test_tokens = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device)
        logits = model(test_tokens[:, :-1])
        assert_real_tensor(logits, name="eval logits")
        logit_mean = logits.float().mean().item()
        logit_std = logits.float().std().item()
        assert -5 <= logit_mean <= 5, f"Logit mean out of range: {logit_mean}"
        assert 0.01 <= logit_std <= 50, f"Logit std out of range: {logit_std}"
    model.train()

    print(f"ok  losses={[f'{loss_i:.4f}' for loss_i in losses]}")


if __name__ == "__main__":
    main()
