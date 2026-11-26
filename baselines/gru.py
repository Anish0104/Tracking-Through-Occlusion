"""CNN-GRU baseline: the conventional recurrent counterpart of the SNN.

Architecture (per frame, then a GRU over time)
    frame [1, 64, 64]
    Conv 3x3, 1 -> 16, pad 1  + ReLU + MaxPool 2      -> [16, 32, 32]   (same channels/pooling
    Conv 3x3, 16 -> 32, pad 1 + ReLU + MaxPool 2      -> [32, 16, 16]    as the SNN encoder)
    flatten                                            -> 8192
    Linear 8192 -> 118 + ReLU   (feature projection)   -> 118
    GRU 118 -> 128 (one layer, unidirectional)         -> h_t [128]      (SNN: 128 recurrent LIF)
    Linear 128 -> 2 + sigmoid   (same readout as SNN)  -> (x, y) in (0, 1)

Why the projection
    The SNN layout in CONTRACT.md (3x3 convs, 8192->128 input weights, 128x128
    recurrence, 128->2 readout) has exactly 1,070,274 parameters. A GRU fed the
    8,192 features directly needs three input matrices (one per gate) and would
    have 3,200,706 parameters (+199%). Projecting to 118 features first keeps
    128 recurrent units and lands at 1,067,064 parameters (-0.30%), well within
    the 10% budget. Breakdown:
        encoder  160 + 4,640        =     4,800
        project  8192*118 + 118     =   966,774
        GRU      3*(118*128 + 128*128 + 2*128) = 95,232
        readout  128*2 + 2          =       258
        total                       = 1,067,064

Causality and state
    The GRU is unidirectional and there is no normalization across time or
    batch, so the output at frame t depends only on frames 0..t of the same
    video. The hidden state starts at zero for every video (reset between
    videos). `step` runs one frame at a time and matches `forward` exactly.
    Inference takes frames only; no labels.
"""

import numpy as np
import torch
from torch import nn

SIZE = 64
SNN_PARAM_BUDGET = 1_070_274     # CONTRACT.md section 4
BUDGET_TOLERANCE = 0.10
CHANNELS = (16, 32)
FEATURES = CHANNELS[-1] * (SIZE // 4) ** 2   # 8192 after two 2x poolings
PROJ_DIM = 118
HIDDEN = 128
EXPECTED_PARAMS = 1_067_064


class CNNGRU(nn.Module):
    def __init__(self, proj_dim=PROJ_DIM, hidden=HIDDEN):
        super().__init__()
        c1, c2 = CHANNELS
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.project = nn.Sequential(nn.Linear(FEATURES, proj_dim), nn.ReLU())
        self.gru = nn.GRU(proj_dim, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.readout = nn.Linear(hidden, 2)
        self.hidden = hidden

    def init_hidden(self, batch_size, device=None):
        """Zero state: used at the start of every video."""
        return torch.zeros(1, batch_size, self.hidden, device=device)

    def _encode(self, frames):
        """frames [B, T, 64, 64] -> features [B, T, proj_dim]; each frame independently."""
        B, T = frames.shape[:2]
        z = self.encoder(frames.reshape(B * T, 1, SIZE, SIZE))
        return self.project(z).reshape(B, T, -1)

    @staticmethod
    def _check(frames, ndim):
        if frames.ndim != ndim or tuple(frames.shape[-2:]) != (SIZE, SIZE):
            want = "[B, T, 64, 64]" if ndim == 4 else "[B, 64, 64]"
            raise ValueError(f"frames must be {want}, got {tuple(frames.shape)}")

    def forward(self, frames, h0=None):
        """frames [B, T, 64, 64] float in [0, 1]. Returns (pred [B, T, 2], h_T [1, B, hidden])."""
        self._check(frames, 4)
        if h0 is None:
            h0 = self.init_hidden(frames.shape[0], frames.device)
        out, h = self.gru(self._encode(frames), h0)
        return torch.sigmoid(self.readout(out)), h

    def step(self, frame, h=None):
        """One frame: frame [B, 64, 64], state h (None = new video). Returns (xy [B, 2], h)."""
        self._check(frame, 3)
        pred, h = self.forward(frame[:, None], h)
        return pred[:, 0], h


def build_model(seed=0, **kwargs):
    """Create a CNN-GRU with reproducible initial weights for a training seed."""
    torch.manual_seed(seed)
    return CNNGRU(**kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_breakdown(model):
    return {name: count_params(m) for name, m in
            (("encoder", model.encoder), ("project", model.project),
             ("gru", model.gru), ("readout", model.readout))}


def within_budget(n_params, budget=SNN_PARAM_BUDGET, tol=BUDGET_TOLERANCE):
    return abs(n_params - budget) <= tol * budget


@torch.no_grad()
def predict(frames, model, batch_size=64, device="cpu"):
    """frames: numpy [N, T, 64, 64] pixels only. Returns pred_xy float32 [N, T, 2].
    Each video starts from a zero hidden state."""
    frames = np.asarray(frames, np.float32)
    if frames.ndim != 4:
        raise ValueError(f"frames must be [N, T, 64, 64], got shape {frames.shape}")
    was_training = model.training
    model.eval().to(device)
    out = []
    for i in range(0, len(frames), batch_size):
        x = torch.from_numpy(frames[i:i + batch_size]).to(device)
        out.append(model(x)[0].cpu().numpy())
    model.train(was_training)
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, frames.shape[1], 2), np.float32)
