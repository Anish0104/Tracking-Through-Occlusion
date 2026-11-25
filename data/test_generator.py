import numpy as np
import pytest

from .generator import TEST_DURATIONS, iter_split, make_batch
from .starter import BARRIER_VALUE, RADIUS, SIZE, T


@pytest.mark.parametrize("motion", ["constant", "smooth"])
def test_pixels_labels_and_motion(motion):
    pixels = np.arange(SIZE) + 0.5
    for duration in TEST_DURATIONS:
        batch = make_batch(3, 42, durations=(duration,), motion=motion)
        assert set(batch) == {"frames", "xy", "visible_frac"}
        assert all(array.dtype == np.float32 for array in batch.values())
        assert batch["frames"].shape == (3, T, SIZE, SIZE)
        for frames, xy, frac in zip(batch["frames"], batch["xy"], batch["visible_frac"]):
            x, y = xy.T * SIZE
            disk = ((pixels[None, None, :] - x[:, None, None]) ** 2
                    + (pixels[None, :, None] - y[:, None, None]) ** 2 <= RADIUS ** 2)
            barrier = frames[0, 0] == BARRIER_VALUE
            expected = np.where(barrier, BARRIER_VALUE, disk).astype(np.float32)
            np.testing.assert_array_equal(frames, expected)
            fraction = (disk & ~barrier).sum((1, 2)) / disk.sum((1, 2))
            np.testing.assert_array_equal(frac, fraction.astype(np.float32))
            assert np.count_nonzero(frac == 0) == duration
            variation = np.ptp(np.diff(x))
            assert variation < 1e-4 if motion == "constant" else variation > 0.1


def test_split_reproducibility():
    first_four = next(iter_split("test", 4))
    singles = iter_split("test", 1)
    for i in range(4):
        video = next(singles)
        for key in first_four:
            np.testing.assert_array_equal(video[key][0], first_four[key][i])
