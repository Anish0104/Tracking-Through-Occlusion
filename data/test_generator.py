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


def test_two_argument_test_bank_matches_every_streamed_video():
    bank = make_batch(1000, 300)
    hidden = (bank["visible_frac"] == 0).sum(axis=1)
    assert set(hidden) == set(TEST_DURATIONS)
    for batch_size in (13, 32):
        offset = 0
        for batch in iter_split("test", batch_size):
            end = offset + len(batch["xy"])
            for key in bank:
                assert np.array_equal(bank[key][offset:end], batch[key])
            offset = end
        assert offset == 1000


def test_custom_batches_keep_training_defaults_and_explicit_overrides():
    custom = make_batch(15, 300)
    explicit = make_batch(15, 300, durations=(0, 2, 4, 8))
    for key in custom:
        np.testing.assert_array_equal(custom[key], explicit[key])
    override = make_batch(1000, 300, durations=(2,))
    assert np.all((override["visible_frac"] == 0).sum(axis=1) == 2)


def test_evaluation_rejects_stale_and_incomplete_predictions(tmp_path):
    from common.io import save_preds
    from .generator import OFFICIAL_TEST_TAG, iter_test_predictions

    path = tmp_path / "preds.npz"
    for tag, n, reason in (("generator_test_s300", 1000, "tag"),
                           (OFFICIAL_TEST_TAG, 999, "shape")):
        save_preds(path, np.zeros((n, T, 2)), "last_pos", 0, tag)
        with pytest.raises(ValueError, match=reason):
            next(iter_test_predictions(path))
