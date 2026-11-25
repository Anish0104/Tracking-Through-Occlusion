"""Starter generator: a tiny stand-in so nobody waits for the full generator.

One bright disk moves at constant velocity past one vertical barrier.
Output follows CONTRACT.md exactly, so code written against this works
unchanged when data/generator.py (the full generator) replaces it.

Usage:
    from data.starter import make_batch
    batch = make_batch(n=16, seed=0)
    batch["frames"]        # float32 [N, T, 64, 64], values in [0, 1]
    batch["xy"]            # float32 [N, T, 2], (x, y) normalized by image size
    batch["visible_frac"]  # float32 [N, T], 1 = fully visible, 0 = fully hidden
"""

import numpy as np

SIZE = 64            # image is SIZE x SIZE pixels
T = 40               # frames per video
RADIUS = 3.0         # disk radius in pixels
DISK_VALUE = 1.0     # disk brightness
BARRIER_VALUE = 0.4  # barrier brightness (darker than the disk, brighter than background)


def _one_video(rng):
    # Motion: start on the left, move right, small vertical drift.
    # Ranges keep the disk inside the frame for all T frames (no bounces).
    x0 = rng.uniform(5, 10)
    y0 = rng.uniform(15, 49)
    vx = rng.uniform(0.6, 1.2)
    vy = rng.uniform(-0.3, 0.3)
    if rng.random() < 0.5:          # half the videos move right-to-left instead
        x0, vx = SIZE - x0, -vx

    # Barrier: a vertical band spanning the full image height.
    bw = rng.uniform(6, 14)
    bx = rng.uniform(26, 38)
    b_left, b_right = bx - bw / 2, bx + bw / 2

    # Pixel centers sit at i + 0.5, so coordinates are continuous in [0, SIZE).
    centers = np.arange(SIZE) + 0.5
    px, py = np.meshgrid(centers, centers)          # px = column (x), py = row (y)
    barrier = (px >= b_left) & (px <= b_right)

    frames = np.zeros((T, SIZE, SIZE), np.float32)
    xy = np.zeros((T, 2), np.float32)
    vis = np.zeros(T, np.float32)
    for t in range(T):
        cx, cy = x0 + vx * t, y0 + vy * t
        disk = (px - cx) ** 2 + (py - cy) ** 2 <= RADIUS ** 2
        visible_pixels = disk & ~barrier
        frames[t][disk] = DISK_VALUE
        frames[t][barrier] = BARRIER_VALUE           # barrier is drawn in front
        xy[t] = (cx / SIZE, cy / SIZE)
        vis[t] = visible_pixels.sum() / max(disk.sum(), 1)
    return frames, xy, vis


def make_batch(n, seed):
    """Return n videos as a dict of numpy arrays (see module docstring)."""
    rng = np.random.default_rng(seed)
    out = [_one_video(rng) for _ in range(n)]
    return {
        "frames": np.stack([o[0] for o in out]),
        "xy": np.stack([o[1] for o in out]),
        "visible_frac": np.stack([o[2] for o in out]),
    }
