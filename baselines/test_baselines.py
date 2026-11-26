"""Tests for the detector and last-position baseline. Run: python -m pytest -q"""

import inspect

import numpy as np
import pytest

from baselines import last_pos
from baselines.detector import SIZE, THRESHOLD, detect
from baselines.predict import run
from common.io import load_preds
from data.starter import BARRIER_VALUE, make_batch


@pytest.fixture(scope="module")
def batch():
    return make_batch(n=32, seed=1)


def blank(n=1, t=1):
    return np.zeros((n, t, SIZE, SIZE), np.float32)


# ---------------------------------------------------------------- detector

def test_single_pixel_uses_pixel_centers():
    f = np.zeros((SIZE, SIZE), np.float32)
    f[10, 20] = 1.0                       # row 10 -> y, column 20 -> x
    xy, n, found = detect(f)
    assert found and n == 1
    assert np.allclose(xy, [(20 + 0.5) / SIZE, (10 + 0.5) / SIZE])


def test_centroid_is_mean_of_bright_pixels():
    f = np.zeros((SIZE, SIZE), np.float32)
    f[0, 0] = f[0, 63] = f[63, 0] = f[63, 63] = 1.0
    xy, n, _ = detect(f)
    assert n == 4 and np.allclose(xy, [0.5, 0.5])


def test_threshold_is_above_barrier():
    assert THRESHOLD > BARRIER_VALUE
    f = np.full((SIZE, SIZE), BARRIER_VALUE, np.float32)      # barrier everywhere, no disk
    xy, n, found = detect(f)
    assert not found and n == 0 and np.isnan(xy).all()


def test_empty_frame_is_no_detection():
    xy, n, found = detect(np.zeros((SIZE, SIZE), np.float32))
    assert not found and np.isnan(xy).all()


def test_detector_matches_starter_visibility(batch):
    _, n, found = detect(batch["frames"])
    vis = batch["visible_frac"]
    assert found.shape == vis.shape
    assert not found[vis == 0].any(), "fully hidden frames must give no detection"
    assert found[vis > 0].all(), "any visible part should be detected"


def test_fully_visible_centroid_is_accurate(batch):
    xy, _, found = detect(batch["frames"])
    full = batch["visible_frac"] == 1
    err_px = np.linalg.norm(xy[full] - batch["xy"][full], axis=-1) * SIZE
    assert err_px.max() < 0.5


def test_partial_centroid_is_visible_part_and_biased(batch):
    xy, n, _ = detect(batch["frames"])
    vis = batch["visible_frac"]
    part = (vis > 0) & (vis < 1)
    assert part.any()
    # the centroid is exactly the mean of the uncovered disk pixels
    i, t = map(int, np.argwhere(part)[0])
    rows, cols = np.nonzero(batch["frames"][i, t] > THRESHOLD)
    assert np.allclose(xy[i, t], [(cols + 0.5).mean() / SIZE, (rows + 0.5).mean() / SIZE])
    assert n[i, t] == len(rows) < n[vis == 1].max()
    err_px = np.linalg.norm(xy[part] - batch["xy"][part], axis=-1) * SIZE
    assert err_px.max() <= 3.0 + 0.5     # bias bounded by about one radius


def test_detector_rejects_bad_input():
    with pytest.raises(ValueError):
        detect(np.zeros((32, 32)))
    with pytest.raises(ValueError):
        detect(np.zeros((SIZE, SIZE)), threshold=0.4)


# ---------------------------------------------------------------- last position

def test_inference_takes_frames_only():
    params = list(inspect.signature(last_pos.predict).parameters)
    assert params[0] == "frames"
    assert not {"xy", "visible_frac", "batch"} & set(params)


def test_shape_dtype_range(batch):
    pred = last_pos.predict(batch["frames"])
    assert pred.shape == batch["xy"].shape and pred.dtype == np.float32
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all()


def test_hand_built_video():
    v = blank(t=6)
    v[0, 1, 10, 20] = 1.0           # first detection at t=1
    v[0, 2, :, 30:40] = BARRIER_VALUE  # t=2: barrier only -> hold
                                    # t=3: empty -> hold
    v[0, 4, 40, 50] = 1.0           # t=4: new detection
                                    # t=5: empty -> hold the new one
    p = last_pos.predict(v)[0]
    a = [(20.5) / SIZE, (10.5) / SIZE]
    b = [(50.5) / SIZE, (40.5) / SIZE]
    assert np.allclose(p, [[0.5, 0.5], a, a, a, b, b])


def test_center_before_first_detection():
    p = last_pos.predict(blank(n=3, t=5))
    assert np.array_equal(p, np.full((3, 5, 2), 0.5, np.float32))


def test_holds_through_occlusion(batch):
    pred = last_pos.predict(batch["frames"])
    _, _, found = detect(batch["frames"])
    hidden = batch["visible_frac"] == 0
    for i, t in zip(*np.nonzero(hidden)):
        prev = np.nonzero(found[i, :t])[0]
        expected = pred[i, prev[-1]] if len(prev) else np.float32([0.5, 0.5])
        assert np.array_equal(pred[i, t], expected)
    # and visible frames use that frame's own detection
    xy, _, _ = detect(batch["frames"])
    assert np.array_equal(pred[found], xy[found])


def test_videos_are_independent(batch):
    frames = batch["frames"]
    together = last_pos.predict(frames)
    alone = np.concatenate([last_pos.predict(frames[i:i + 1]) for i in range(len(frames))])
    assert np.array_equal(together, alone)
    # reordering the batch reorders the predictions and nothing else
    perm = np.random.default_rng(0).permutation(len(frames))
    assert np.array_equal(last_pos.predict(frames[perm]), together[perm])


@pytest.mark.parametrize("t_cut", [0, 7, 20, 39])
def test_causal_future_frames_do_not_matter(batch, t_cut):
    frames = batch["frames"]
    noisy = frames.copy()
    noisy[:, t_cut + 1:] = np.random.default_rng(t_cut).random(noisy[:, t_cut + 1:].shape)
    a, b = last_pos.predict(frames), last_pos.predict(noisy)
    assert np.array_equal(a[:, :t_cut + 1], b[:, :t_cut + 1])
    # truncating the video gives the same prefix
    assert np.array_equal(last_pos.predict(frames[:, :t_cut + 1]), a[:, :t_cut + 1])


def test_deterministic(batch):
    assert np.array_equal(last_pos.predict(batch["frames"]), last_pos.predict(batch["frames"]))


def test_input_is_not_modified(batch):
    frames = batch["frames"].copy()
    last_pos.predict(frames)
    assert np.array_equal(frames, batch["frames"])


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        last_pos.predict(np.zeros((40, SIZE, SIZE)))


# ---------------------------------------------------------------- prediction file

def test_cli_writes_contract_file(tmp_path):
    path = run(model="last_pos", n=8, data_seed=200, out_dir=tmp_path)   # dev seed; never 300
    assert path.name == "last_pos_seed0.npz"
    out = load_preds(path)
    assert out["model"] == "last_pos" and out["seed"] == 0
    assert out["data_tag"] == "starter_test_s200"
    assert out["pred_xy"].shape == (8, 40, 2) and out["pred_xy"].dtype == np.float32
    assert np.array_equal(out["pred_xy"], last_pos.predict(make_batch(8, 200)["frames"]))


# ================================================================ Kalman (phase 2)

from dataclasses import replace  # noqa: E402

from baselines import kalman  # noqa: E402
from baselines.dev_metrics import check_not_test, pixel_error  # noqa: E402
from baselines.kalman import F, KalmanConfig, measurement_std_px, track  # noqa: E402

SHARP = KalmanConfig(accel_px=0.001, meas_px=0.05, partial_px=4.0, min_pixels=1)


def dets(points, n_px=28):
    """points: list over time of (x, y) or None -> one-video detector outputs."""
    T = len(points)
    xy = np.full((1, T, 2), np.nan, np.float32)
    n = np.zeros((1, T), np.int64)
    for t, p in enumerate(points):
        if p is not None:
            xy[0, t] = p
            n[0, t] = n_px[t] if isinstance(n_px, list) else n_px
    return xy, n, n > 0


def line(t, x0=0.2, y0=0.5, vx=0.01, vy=-0.004):
    return (x0 + vx * t, y0 + vy * t)


def test_kalman_constant_velocity_prediction():
    pts = [line(t) for t in range(12)] + [None] * 10
    pred, st = track(*dets(pts), SHARP, return_state=True)
    truth = np.array([line(t) for t in range(22)])
    assert np.abs(pred[0] - truth)[3:].max() < 1e-3          # after a short warm-up
    assert np.allclose(st["state"][0, 11, 2:], [0.01, -0.004], atol=2e-4)


def test_kalman_update_moves_toward_measurement():
    cfg = KalmanConfig(accel_px=0.01, meas_px=1.0, partial_px=0.0)
    pred, st = track(*dets([(0.5, 0.5), (0.6, 0.5)]), cfg, return_state=True)
    prior_x = 0.5                                             # zero initial velocity
    assert prior_x < pred[0, 1, 0] < 0.6                      # between prior and measurement
    prior_var = (F @ st["cov"][0, 0] @ F.T)[0, 0]
    assert st["cov"][0, 1, 0, 0] < prior_var                  # update reduces uncertainty
    tight = track(*dets([(0.5, 0.5), (0.6, 0.5)]), replace(cfg, meas_px=1e-4))
    assert abs(tight[0, 1, 0] - 0.6) < 1e-6                   # tiny R -> follow measurement


def test_kalman_occlusion_is_prediction_only():
    pts = [line(t) for t in range(8)] + [None] * 6
    _, st = track(*dets(pts), SHARP, return_state=True)
    s, P = st["state"][0], st["cov"][0]
    for t in range(8, 14):
        assert not st["updated"][0, t]
        assert np.allclose(s[t], F @ s[t - 1], atol=1e-12)
        assert np.trace(P[t]) > np.trace(P[t - 1])


def test_kalman_partial_noise_model():
    cfg = KalmanConfig(meas_px=0.3, partial_px=4.0, full_px=28.0)
    std = measurement_std_px(np.arange(0, 40), cfg)
    assert np.all(np.diff(std) <= 0)                          # fewer pixels -> noisier
    assert np.isclose(std[28], 0.3) and np.isclose(std[39], 0.3)
    assert np.isclose(std[0], np.hypot(0.3, 4.0))
    flat = measurement_std_px(np.arange(1, 40), replace(cfg, partial_px=0.0))
    assert np.allclose(flat, 0.3)


def test_kalman_partial_detection_gets_less_weight():
    pts = [line(t) for t in range(6)] + [(0.40, 0.5)]        # last one is off-track
    full = track(*dets(pts, n_px=28), SHARP)[0, -1, 0]
    part = track(*dets(pts, n_px=[28] * 6 + [5]), SHARP)[0, -1, 0]
    expected = line(6)[0]
    assert abs(part - expected) < abs(full - expected)


def test_kalman_min_pixels_gate():
    pts = [line(t) for t in range(5)] + [(0.9, 0.9)]
    _, st = track(*dets(pts, n_px=[28] * 5 + [3]), replace(SHARP, min_pixels=4), return_state=True)
    assert not st["updated"][0, 5]                             # sliver ignored
    assert np.allclose(st["state"][0, 5], F @ st["state"][0, 4])


def test_kalman_recovery_after_gap():
    # The object changes velocity while hidden; a filter whose process noise
    # allows manoeuvres must lock on again. (The starter-tuned accel_px is tiny
    # because starter motion is exactly constant-velocity, so it re-locks
    # slowly after a manoeuvre; that is a tuning issue, not a filter bug.)
    pts = [line(t) for t in range(10)] + [None] * 5 + \
          [line(t, x0=0.25, vx=0.02) for t in range(15, 25)]
    pred, st = track(*dets(pts), replace(SHARP, accel_px=0.1), return_state=True)
    truth = np.array([line(t, x0=0.25, vx=0.02) for t in range(15, 25)])
    prior = (F @ st["state"][0, 14])[:2]                       # coasted guess before the update
    assert np.linalg.norm(prior - truth[0]) * SIZE > 1.0       # the gap really misled it
    err_px = np.linalg.norm(pred[0, 15:] - truth, axis=-1) * SIZE
    assert err_px[-3:].max() < 0.5                             # recovered within the window


def test_kalman_delayed_first_detection():
    pts = [None] * 5 + [line(t) for t in range(5, 10)]
    pred, st = track(*dets(pts), SHARP, return_state=True)
    assert np.array_equal(pred[0, :5], np.full((5, 2), 0.5, np.float32))
    assert not st["active"][0, :5].any() and st["active"][0, 5:].all()
    assert np.allclose(pred[0, 5], line(5), atol=1e-6)         # starts at the measurement
    assert np.allclose(st["state"][0, 5, 2:], 0.0)             # with zero velocity


def test_kalman_never_detected_video():
    pred = kalman.predict(blank(n=2, t=6))
    assert np.array_equal(pred, np.full((2, 6, 2), 0.5, np.float32))


def test_kalman_clipping_policy():
    pts = [line(t, x0=0.80, vx=0.03) for t in range(6)] + [None] * 10
    pred, st = track(*dets(pts), SHARP, return_state=True)
    assert st["state"][0, -1, 0] > 1.0                         # state is not clipped
    assert pred[0, -1, 0] == 1.0 and (pred >= 0).all() and (pred <= 1).all()


def test_kalman_beats_last_pos_on_hidden_starter_frames(batch):
    hidden = batch["visible_frac"] == 0
    ek = pixel_error(kalman.predict(batch["frames"]), batch["xy"])
    el = pixel_error(last_pos.predict(batch["frames"]), batch["xy"])
    assert ek[hidden].mean() < 1.0 < el[hidden].mean()
    assert (ek[hidden] <= 5).all()


def test_kalman_recovers_when_disk_reappears(batch):
    err = pixel_error(kalman.predict(batch["frames"]), batch["xy"])
    vis = batch["visible_frac"]
    checked = 0
    for i in range(len(vis)):
        h = np.nonzero(vis[i] == 0)[0]
        back = np.nonzero(vis[i] == 1)[0]
        back = back[back > h[-1]] if len(h) else []
        if len(back) >= 3:
            assert err[i, back[:3]].max() <= 5.0
            checked += 1
    assert checked > 0


def test_kalman_shape_dtype_finite(batch):
    pred = kalman.predict(batch["frames"])
    assert pred.shape == batch["xy"].shape and pred.dtype == np.float32
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all()


def test_kalman_inference_takes_frames_only():
    for fn in (kalman.predict, kalman.track):
        assert not {"xy_true", "visible_frac", "batch"} & set(inspect.signature(fn).parameters)
    assert list(inspect.signature(kalman.predict).parameters)[0] == "frames"


def test_kalman_videos_are_independent(batch):
    frames = batch["frames"]
    together = kalman.predict(frames)
    alone = np.concatenate([kalman.predict(frames[i:i + 1]) for i in range(len(frames))])
    assert np.allclose(together, alone, atol=1e-7)
    perm = np.random.default_rng(0).permutation(len(frames))
    assert np.allclose(kalman.predict(frames[perm]), together[perm], atol=1e-7)


@pytest.mark.parametrize("t_cut", [0, 7, 20, 39])
def test_kalman_causal(batch, t_cut):
    frames = batch["frames"]
    noisy = frames.copy()
    noisy[:, t_cut + 1:] = np.random.default_rng(t_cut).random(noisy[:, t_cut + 1:].shape)
    a, b = kalman.predict(frames), kalman.predict(noisy)
    assert np.array_equal(a[:, :t_cut + 1], b[:, :t_cut + 1])
    assert np.array_equal(kalman.predict(frames[:, :t_cut + 1]), a[:, :t_cut + 1])


def test_kalman_deterministic_and_input_untouched(batch):
    frames = batch["frames"].copy()
    assert np.array_equal(kalman.predict(frames), kalman.predict(frames))
    assert np.array_equal(frames, batch["frames"])


def test_kalman_config_file_is_from_validation():
    import json
    doc = json.loads(kalman.CONFIG_PATH.read_text())
    assert doc["selection"]["data_seed"] == 200
    assert doc["selection"]["module"] == "data.generator"
    assert doc["selection"]["objective"].startswith("overall mean pixel error")
    cfg = kalman.load_config()
    assert kalman.default_config() == cfg
    assert cfg.threshold > BARRIER_VALUE


def test_kalman_config_validation():
    with pytest.raises(ValueError):
        KalmanConfig(meas_px=0)
    with pytest.raises(ValueError):
        KalmanConfig(min_pixels=0)
    with pytest.raises(ValueError):
        KalmanConfig(partial_px=-1)


def test_dev_metrics_refuses_test_bank():
    with pytest.raises(SystemExit):
        check_not_test(300)
    check_not_test(200)


def test_kalman_cli_writes_contract_file(tmp_path):
    # the installed config is tuned on data.generator, so predict on that generator
    from data.generator import make_batch as generator_batch
    path = run(model="kalman", data="generator", n=8, data_seed=200, out_dir=tmp_path)   # never 300
    assert path.name == "kalman_seed0.npz"
    out = load_preds(path)
    assert out["model"] == "kalman" and out["seed"] == 0
    assert out["data_tag"] == "generator_test_s200"
    assert out["pred_xy"].shape == (8, 40, 2) and out["pred_xy"].dtype == np.float32
    assert np.array_equal(out["pred_xy"], kalman.predict(generator_batch(8, 200)["frames"]))
