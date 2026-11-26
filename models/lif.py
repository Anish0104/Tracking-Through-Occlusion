"""Leaky integrate-and-fire (LIF) neuron and fast-sigmoid surrogate gradient."""

import torch
import torch.nn as nn


class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside step forward, fast-sigmoid derivative backward.

    Forward:  S(u) = 1 if u >= 0 else 0   (where u = mem - threshold)
    Backward: dS/du ≈ 1 / (1 + slope * |u|)^2
    """

    @staticmethod
    def forward(ctx, mem_minus_thr, slope):
        ctx.save_for_backward(mem_minus_thr)
        ctx.slope = slope
        return (mem_minus_thr >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (mem_minus_thr,) = ctx.saved_tensors
        surrogate_grad = 1.0 / (1.0 + ctx.slope * mem_minus_thr.abs()) ** 2
        return grad_output * surrogate_grad, None


class LIF(nn.Module):
    """Leaky integrate-and-fire neuron(s) with fixed decay and threshold."""

    def __init__(self, beta=0.9, threshold=1.0, slope=25.0):
        super().__init__()
        self.beta = float(beta)            # membrane decay factor per timestep
        self.threshold = float(threshold)  # firing threshold
        self.slope = float(slope)          # surrogate gradient steepness

    def forward(self, current, mem):
        mem = self.beta * mem + current
        spk = FastSigmoidSpike.apply(mem - self.threshold, self.slope)
        mem = mem - spk * self.threshold   # reset by subtraction
        return spk, mem


def _demo():
    lif = LIF(beta=0.9, threshold=1.0, slope=25.0)

    print("1) Constant input -> spike train (beta=0.9, threshold=1.0, I=0.25):")
    mem = torch.zeros(1)
    current = torch.tensor([0.25])
    spikes = []
    for t in range(15):
        spk, mem = lif(current, mem)
        spikes.append(int(spk.item()))
        print(f"   t={t:2d}  spk={int(spk.item())}  mem_after={mem.item():.4f}")
    print(f"   spike train: {spikes} (total={sum(spikes)})")

    print("\n2) Gradient flow through surrogate spike:")
    learnable_current = torch.tensor([0.25], requires_grad=True)
    mem = torch.zeros(1)
    total_spikes = torch.zeros(1)
    for _ in range(15):
        spk, mem = lif(learnable_current, mem)
        total_spikes = total_spikes + spk

    # Target 4 spikes in 15 steps; without a surrogate, d(total_spikes)/dI would be 0.
    loss = (total_spikes - 4.0) ** 2
    loss.backward()
    print(f"   total_spikes={total_spikes.item():.1f}, loss={loss.item():.4f}, "
          f"d(loss)/d(current)={learnable_current.grad.item():.4f}")


if __name__ == "__main__":
    _demo()
