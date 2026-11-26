"""Recurrent spiking neural network for tracking a moving disk through occlusion."""

import torch
import torch.nn as nn

from models.lif import LIF

SIZE = 64
EXPECTED_PARAMS = 1_070_274


class RecurrentSNN(nn.Module):
    """2-layer spiking CNN encoder -> 128-neuron recurrent LIF -> (x, y) readout."""

    def __init__(self, beta=0.9, threshold=1.0, slope=25.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1 = LIF(beta=beta, threshold=threshold, slope=slope)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = LIF(beta=beta, threshold=threshold, slope=slope)
        self.pool2 = nn.MaxPool2d(2)

        self.fc_in = nn.Linear(32 * 16 * 16, 128)
        self.fc_rec = nn.Linear(128, 128)
        self.lif3 = LIF(beta=beta, threshold=threshold, slope=slope)

        self.readout = nn.Linear(128, 2)

    def init_state(self, batch_size, device=None):
        """Zero all membrane potentials and previous recurrent spikes for a new video."""
        return (
            torch.zeros(batch_size, 16, SIZE, SIZE, device=device),
            torch.zeros(batch_size, 32, SIZE // 2, SIZE // 2, device=device),
            torch.zeros(batch_size, 128, device=device),
            torch.zeros(batch_size, 128, device=device),
        )

    def step(self, frame, state=None):
        """Process one frame [B, 1, 64, 64] (or [B, 64, 64]).

        Returns (xy [B, 2], next_state, total_frame_spikes [B]).
        """
        if frame.ndim == 3:
            frame = frame.unsqueeze(1)
        if frame.ndim != 4 or frame.shape[1:] != (1, SIZE, SIZE):
            raise ValueError(f"frame must be [B, 1, 64, 64] or [B, 64, 64], got {tuple(frame.shape)}")

        if state is None:
            state = self.init_state(frame.shape[0], frame.device)
        mem1, mem2, mem3, spk3 = state

        spk1, mem1 = self.lif1(self.conv1(frame), mem1)
        p1 = self.pool1(spk1)

        spk2, mem2 = self.lif2(self.conv2(p1), mem2)
        p2 = self.pool2(spk2)

        flat = p2.flatten(1)
        cur3 = self.fc_in(flat) + self.fc_rec(spk3)
        spk3, mem3 = self.lif3(cur3, mem3)

        xy = torch.sigmoid(self.readout(mem3))
        frame_spikes = spk1.sum(dim=(1, 2, 3)) + spk2.sum(dim=(1, 2, 3)) + spk3.sum(dim=1)
        return xy, (mem1, mem2, mem3, spk3), frame_spikes

    def forward(self, frames, state=None, reset_each_frame=False):
        """Process a video batch [B, T, 1, 64, 64] (or [B, T, 64, 64]).

        Returns (pred_xy [B, T, 2], spikes [B, T]).
        """
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5 or frames.shape[2:] != (1, SIZE, SIZE):
            raise ValueError(
                f"frames must be [B, T, 1, 64, 64] or [B, T, 64, 64], got {tuple(frames.shape)}"
            )

        B, T = frames.shape[:2]
        if state is None:
            state = self.init_state(B, frames.device)

        preds = []
        spikes = []
        for t in range(T):
            if reset_each_frame:
                state = self.init_state(B, frames.device)
            xy_t, state, spk_t = self.step(frames[:, t], state)
            preds.append(xy_t)
            spikes.append(spk_t)

        return torch.stack(preds, dim=1), torch.stack(spikes, dim=1)


def build_model(seed=0, beta=0.9, threshold=1.0, slope=25.0):
    """Create a RecurrentSNN with reproducible initial weights for a training seed."""
    torch.manual_seed(seed)
    return RecurrentSNN(beta=beta, threshold=threshold, slope=slope)


def count_params(model):
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_breakdown(model):
    """Trainable parameter count per layer."""
    layers = (
        ("conv1", model.conv1),
        ("lif1", model.lif1),
        ("conv2", model.conv2),
        ("lif2", model.lif2),
        ("fc_in", model.fc_in),
        ("fc_rec", model.fc_rec),
        ("lif3", model.lif3),
        ("readout", model.readout),
    )
    return {name: count_params(layer) for name, layer in layers}


def _demo():
    model = build_model(seed=0)
    breakdown = param_breakdown(model)
    total = count_params(model)

    print("Parameter breakdown:")
    for name, n in breakdown.items():
        print(f"  {name:<8s}: {n:>10,}")
    print(f"  {'total':<8s}: {total:>10,}")
    assert total == EXPECTED_PARAMS, f"Expected {EXPECTED_PARAMS:,}, got {total:,}"
    print(f"Verified total == {EXPECTED_PARAMS:,}")

    x = torch.rand(2, 40, 1, 64, 64)
    pred_xy, spikes = model(x)
    print(f"\nForward pass over {tuple(x.shape)}:")
    print(f"  pred_xy: shape={tuple(pred_xy.shape)}, range=[{pred_xy.min().item():.4f}, {pred_xy.max().item():.4f}]")
    print(f"  spikes:  shape={tuple(spikes.shape)}, mean_per_frame={spikes.mean().item():.1f}")


if __name__ == "__main__":
    _demo()
