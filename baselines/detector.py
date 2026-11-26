"""Centroid detector: finds the disk from pixels only.

Shared by the last-position and Kalman baselines. It never sees labels.

How it works
    A pixel belongs to the disk when its value is above THRESHOLD. The disk is
    1.0, the barrier 0.4 and the background 0 (CONTRACT.md section 1), so 0.7,
    halfway between disk and barrier, separates them with margin to spare if
    the full generator adds noise or anti-aliasing.
    The estimate is the mean position of those pixels. Pixel (row i, col j)
    has its center at (j + 0.5, i + 0.5), and both are divided by 64, the
    same units as the `xy` labels.

Visibility cases
    fully visible   all disk pixels are bright, so the centroid is the disk center
                    up to pixel discretisation (well under 1 px).
    partial         the barrier is drawn over the disk, so only the uncovered
                    part is bright. The centroid is the centroid of that visible
                    part and is biased toward the uncovered side (at most about one
                    disk radius, 3 px, in x for a vertical barrier). It is still
                    returned, together with the pixel count, so callers can decide
                    how much to trust it (Kalman: scale noise by pixel count).
    fully hidden /  fewer than `min_pixels` bright pixels: no detection; the
    no disk         centroid is NaN and `found` is False.
"""

import numpy as np

SIZE = 64
THRESHOLD = 0.7     # strictly between barrier (0.4) and disk (1.0)
MIN_PIXELS = 1      # bright pixels needed to count as a detection

_CENTERS = np.arange(SIZE, dtype=np.float64) + 0.5     # pixel i has its center at i + 0.5


def detect(frames, threshold=THRESHOLD, min_pixels=MIN_PIXELS):
    """Detect the disk in each frame independently.

    frames: array [..., 64, 64] (any leading dims, e.g. [N] or [N, T]).
    Returns (xy, n_pixels, found):
        xy        float32 [..., 2]  normalized (x, y) centroid; NaN where not found
        n_pixels  int64   [...]     number of bright pixels
        found     bool    [...]     n_pixels >= min_pixels
    """
    frames = np.asarray(frames)
    if frames.ndim < 2 or frames.shape[-2:] != (SIZE, SIZE):
        raise ValueError(f"frames must have shape [..., {SIZE}, {SIZE}], got {frames.shape}")
    if not threshold > 0.4:
        raise ValueError("threshold must be above the barrier brightness 0.4")
    if min_pixels < 1:
        raise ValueError("min_pixels must be at least 1")

    mask = frames > threshold                       # [..., H, W]
    n = mask.sum(axis=(-2, -1))                     # [...]
    found = n >= min_pixels

    # x is the column index, y is the row index.
    col_sum = (mask.sum(axis=-2) * _CENTERS).sum(axis=-1)   # sum over bright pixels of (j + 0.5)
    row_sum = (mask.sum(axis=-1) * _CENTERS).sum(axis=-1)   # sum over bright pixels of (i + 0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = col_sum / n / SIZE
        y = row_sum / n / SIZE
    xy = np.stack([x, y], axis=-1)
    xy[~found] = np.nan
    return xy.astype(np.float32), n.astype(np.int64), found
