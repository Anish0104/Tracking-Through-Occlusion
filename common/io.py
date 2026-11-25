"""Prediction files: the only thing models hand to evaluation.

Every model saves  preds/<model>_seed<k>.npz  with:
    pred_xy   float32 [N, T, 2]   predicted (x, y), normalized like the labels
    model     str                 e.g. "snn", "snn_reset", "gru", "kalman", "last_pos"
    seed      int                 training seed (0 for non-learned baselines)
    data_tag  str                 which test bank was used, e.g. "starter_test_s2026"
    spikes    float32 [N, T]      optional: total spikes per frame (SNN only)
    n_params  int                 optional: trainable parameter count
"""

import numpy as np


def save_preds(path, pred_xy, model, seed, data_tag, spikes=None, n_params=None):
    pred_xy = np.asarray(pred_xy, np.float32)
    assert pred_xy.ndim == 3 and pred_xy.shape[-1] == 2, "pred_xy must be [N, T, 2]"
    extra = {}
    if spikes is not None:
        spikes = np.asarray(spikes, np.float32)
        assert spikes.shape == pred_xy.shape[:2], "spikes must be [N, T]"
        extra["spikes"] = spikes
    if n_params is not None:
        extra["n_params"] = np.int64(n_params)
    np.savez_compressed(path, pred_xy=pred_xy, model=model, seed=np.int64(seed),
                        data_tag=data_tag, **extra)


def load_preds(path):
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    for k in ("model", "data_tag"):
        out[k] = str(out[k])
    out["seed"] = int(out["seed"])
    return out
