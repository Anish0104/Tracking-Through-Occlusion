"""Fake predictions for building evaluation before real models exist.

Noise grows while the target is hidden, roughly like a real tracker would behave.
"""

import numpy as np


def fake_preds(xy, visible_frac, seed=0, visible_px=1.0, hidden_px=4.0, size=64):
    rng = np.random.default_rng(seed)
    sigma = np.where(visible_frac > 0, visible_px, hidden_px) / size   # [N, T]
    noise = rng.normal(size=xy.shape) * sigma[..., None]
    return (xy + noise).astype(np.float32)
