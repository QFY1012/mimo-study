"""LoRA from scratch: a CPU-sized demo of the ideas in notes/lora-tutorial.md.

Run: python examples/lora_demo.py
"""

import math

import torch
import torch.nn as nn

torch.manual_seed(0)


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B (A x), with W frozen and only A, B trained."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        # A: random, B: zero -> the adapter starts as an exact no-op.
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        # x -> A (down to r dims) -> B (back up). Never materialize B @ A while training.
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)

    def delta_weight(self):
        return self.scale * (self.B @ self.A)


def section(title):
    print(f"\n=== {title} ===")


d_in, d_out, n = 256, 256, 4096
base = nn.Linear(d_in, d_out, bias=False)

section("1. Parameter count")
for r in (1, 4, 16, 64):
    lora = r * (d_in + d_out)
    print(f"r={r:>2}: LoRA params {lora:>7,}  vs full {d_in * d_out:,}  ({lora / (d_in * d_out):.1%})")

section("2. B = 0 at init, so the model is unchanged")
layer = LoRALinear(base, r=8, alpha=8)
x = torch.randn(32, d_in)
print("max |lora(x) - base(x)| =", (layer(x) - base(x)).abs().max().item())

section("3. First step: only B gets a gradient (A's gradient is multiplied by B = 0)")
layer(x).pow(2).mean().backward()
print(f"|grad A| = {layer.A.grad.norm():.3e}   |grad B| = {layer.B.grad.norm():.3e}")
print("base W grad:", base.weight.grad)

section("4. Fit a target whose change is rank 4: r < 4 cannot, r >= 4 can")
true_rank = 4
delta_true = 0.5 * torch.randn(d_out, true_rank) @ torch.randn(true_rank, d_in) / math.sqrt(d_in)
X = torch.randn(n, d_in)
Y = X @ (base.weight + delta_true).T
for r in (1, 2, 4, 16):
    layer = LoRALinear(base, r=r, alpha=r)
    opt = torch.optim.Adam([layer.A, layer.B], lr=1e-2)
    for _ in range(1500):
        opt.zero_grad()
        loss = (layer(X) - Y).pow(2).mean()
        loss.backward()
        opt.step()
    print(f"r={r:>2}: final MSE {loss.item():.2e}")

section("5. Merge: W' = W + (alpha/r) B A gives the same output with zero extra cost")
merged = base.weight + layer.delta_weight()
print("fp32 max |merged - unmerged| =", (X[:8] @ merged.T - layer(X[:8])).abs().max().item())

section("6. Merging a tiny update into bf16 weights rounds it away")
W = torch.randn(d_out, d_in) * 0.02
for size in (1e-3, 1e-4, 1e-5, 1e-6):
    dW = torch.full_like(W, size)
    W_bf16 = W.to(torch.bfloat16)
    lost = ((W_bf16 + dW.to(torch.bfloat16)) == W_bf16).float().mean().item()
    print(f"update {size:.0e}: {lost:6.1%} of elements unchanged after bf16 merge")
