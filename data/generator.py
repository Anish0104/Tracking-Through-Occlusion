"""Generate labeled videos and stream reproducible, balanced dataset splits."""

from itertools import product

import numpy as np

from .starter import BARRIER_VALUE, DISK_VALUE, RADIUS, SIZE, T

TRAIN_DURATIONS = (0, 2, 4, 8)
TEST_DURATIONS = (*TRAIN_DURATIONS, 12)
SPLITS = {
    "train": (5000, 100, TRAIN_DURATIONS),
    "val": (1000, 200, TRAIN_DURATIONS),
    "test": (1000, 300, TEST_DURATIONS),
}
OFFICIAL_TEST_TAG = "generator_test_s300_v1"

# Mean horizontal displacement in pixels per frame.
SPEED_BINS = ((0.72, 0.87), (0.95, 1.10), (1.18, 1.32))
MARGIN = 4
_OFFSETS = np.arange(-3, 4)


def _trajectory(rng, speed_bin, motion):
    speed = rng.uniform(*SPEED_BINS[speed_bin])
    increments = np.full(T - 1, speed)
    if motion == "smooth":
        phase = rng.uniform(0, 2 * np.pi)
        wave = np.sin(np.arange(T - 1) * 2 * np.pi / (T - 1) + phase)
        increments *= 1 + rng.uniform(0.12, 0.28) * wave
    x = np.r_[0., np.cumsum(increments)]
    x += rng.uniform(RADIUS, SIZE - RADIUS - x[-1])
    if rng.integers(2):
        x = SIZE - x
    y = rng.uniform(16, 48) + np.arange(T) * rng.uniform(-0.25, 0.25)
    return np.column_stack((x, y)).astype(np.float32)


def _video(rng, speed_bin, duration, motion):
    for _ in range(256):
        xy = _trajectory(rng, speed_bin, motion)
        cols = np.floor(xy[:, 0]).astype(int)[:, None] + _OFFSETS
        rows = np.floor(xy[:, 1]).astype(int)[:, None] + _OFFSETS
        dx = cols + 0.5 - xy[:, :1]
        dy = rows + 0.5 - xy[:, 1:]
        disk = dy[:, :, None] ** 2 + dx[:, None, :] ** 2 <= RADIUS ** 2
        occupied = disk.any(axis=1)
        if duration:
            start = rng.integers(10, T - duration - 9)
            window = slice(start, start + duration)
            covered = cols[window][occupied[window]]
            left, right = covered.min(), covered.max()
        else:
            # Place a visible barrier entirely outside the disk's swept area.
            first, last = cols[occupied].min(), cols[occupied].max()
            low, high = (0, first) if first >= SIZE - 1 - last else (last + 1, SIZE)
            width = rng.integers(2, min(9, high - low) + 1)
            left = rng.integers(low, high - width + 1)
            right = left + width - 1
        behind = (cols >= left) & (cols <= right)
        visible = disk & ~behind[:, None, :]
        frac = visible.sum(axis=(1, 2)) / disk.sum(axis=(1, 2))
        if duration and (np.count_nonzero(frac == 0) != duration
                         or np.any(frac[:MARGIN] != 1) or np.any(frac[-MARGIN:] != 1)):
            continue
        frames = np.zeros((T, SIZE, SIZE), np.float32)
        frames[:, :, left:right + 1] = BARRIER_VALUE
        t, i, j = np.nonzero(visible)
        frames[t, rows[t, i], cols[t, j]] = DISK_VALUE
        return frames, xy / SIZE, frac.astype(np.float32)
    raise RuntimeError(f"Could not generate duration {duration} in speed bin {speed_bin}")


def _samples(n, seed, durations, motion):
    if n < 1 or motion not in ("mixed", "constant", "smooth"):
        raise ValueError("Use a positive count and mixed, constant, or smooth motion")
    durations = tuple(durations)
    if not durations or len(set(durations)) != len(durations) or any(
            d not in TEST_DURATIONS for d in durations):
        raise ValueError("Use distinct durations from 0, 2, 4, 8, 12")
    rng = np.random.default_rng(seed)
    cells = list(product(range(len(SPEED_BINS)), durations))
    order = np.resize(rng.permutation(len(cells)), n)
    rng.shuffle(order)
    for index in order:
        speed_bin, duration = cells[index]
        mode = rng.choice(("constant", "smooth")) if motion == "mixed" else motion
        yield _video(rng, speed_bin, duration, mode)


def _pack(samples):
    return dict(zip(("frames", "xy", "visible_frac"), map(np.stack, zip(*samples))))


def make_batch(n, seed, *, durations=None, motion="mixed"):
    """Return an official bank for a matching size/seed or generate a custom batch."""
    if durations is None:
        durations = TRAIN_DURATIONS
        if motion == "mixed":
            for size, split_seed, split_durations in SPLITS.values():
                if (n, seed) == (size, split_seed):
                    durations = split_durations
                    break
    return _pack(_samples(n, seed, durations, motion))


def iter_split(split, batch_size=16):
    """Yield a fixed split in order, independent of the requested batch size."""
    from itertools import islice

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n, seed, durations = SPLITS[split]
    samples = _samples(n, seed, durations, "mixed")
    while batch := list(islice(samples, batch_size)):
        yield _pack(batch)


def iter_test_predictions(prediction_file, batch_size=16):
    """Pair saved prediction rows with the official labels in dataset order."""
    from common.io import load_preds

    saved = load_preds(prediction_file)
    if saved["data_tag"] != OFFICIAL_TEST_TAG:
        raise ValueError(f"Expected official test data tag {OFFICIAL_TEST_TAG}")
    predictions = saved["pred_xy"]
    if predictions.shape != (SPLITS["test"][0], T, 2):
        raise ValueError("Prediction shape does not match the official test bank")
    offset = 0
    for batch in iter_split("test", batch_size):
        end = offset + len(batch["xy"])
        yield predictions[offset:end], batch["xy"], batch["visible_frac"]
        offset = end
