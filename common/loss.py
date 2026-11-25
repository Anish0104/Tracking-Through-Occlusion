"""Shared training loss. Both neural models (SNN and CNN-GRU) must use this.

Per sequence: mean squared error on visible frames and on fully hidden
frames, averaged with equal weight. Sequences with no hidden frame use the
visible part only. The batch loss is the mean over sequences.
This stops the many visible frames from drowning out the few hidden ones.
"""

import torch


def balanced_mse(pred, target, visible_frac):
    """pred, target: [B, T, 2]; visible_frac: [B, T]. Returns a scalar tensor."""
    se = ((pred - target) ** 2).sum(dim=-1)          # squared error per frame [B, T]
    hidden = (visible_frac <= 0).float()
    visible = 1.0 - hidden

    n_vis = visible.sum(dim=1)
    n_hid = hidden.sum(dim=1)
    vis_mean = (se * visible).sum(dim=1) / n_vis.clamp(min=1)
    hid_mean = (se * hidden).sum(dim=1) / n_hid.clamp(min=1)

    has_hidden = n_hid > 0
    per_seq = torch.where(has_hidden, 0.5 * (vis_mean + hid_mean), vis_mean)
    return per_seq.mean()
