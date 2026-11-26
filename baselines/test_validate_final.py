"""Tests for baselines/validate_final.py (metric helpers; no data generation)."""

import inspect
import re

import numpy as np

from baselines import validate_final as vf

T = 40


def _video(first_hidden=None, d=0):
    vis = np.ones(T, np.float32)
    if d:
        vis[first_hidden - 1] = vis[first_hidden + d] = 0.5          # partial entry / exit frames
        vis[first_hidden:first_hidden + d] = 0
    return vis


def _bank():
    vis = np.stack([_video(), _video(15, 2), _video(12, 4), _video(20, 8)])
    xy = np.stack([np.column_stack([np.linspace(0.1, 0.9, T), np.full(T, 0.5)])] * 4).astype(np.float32)
    durations = (vis == 0).sum(1)
    return xy, vis, durations, vf.hidden_spans(vis)


def test_pixel_error_is_normalized_difference_times_64():
    a = np.zeros((1, 1, 2))
    assert vf.pixel_error(a + [3 / 64, 4 / 64], a)[0, 0] == 5.0


def test_hidden_spans_and_durations():
    xy, vis, durations, spans = _bank()
    assert durations.tolist() == [0, 2, 4, 8]
    assert spans == [None, (15, 16), (12, 15), (20, 27)]


def test_recovery_follows_contract_definition():
    xy, vis, durations, spans = _bank()
    err = np.zeros((4, T))
    assert np.isnan(vf.recovery(err, vis, spans)[0])                 # no occlusion
    assert vf.recovery(err, vis, spans)[1:].tolist() == [0, 0, 0]
    # duration 8: hidden 20..27, partial 28, full reappearance t0 = 29
    err[3, 29:31] = 6.0                                              # two bad frames after t0
    assert vf.recovery(err, vis, spans)[3] == 2
    err[3, 33] = 6.0                                                  # breaks the run 31..33; next is 34..36
    assert vf.recovery(err, vis, spans)[3] == 5
    err[3, 29:] = 6.0
    err[3, 25:29] = 0.0                                               # good hidden/partial frames do not count
    assert vf.recovery(err, vis, spans)[3] == np.inf                 # never recovered


def test_metrics_perfect_and_constant_offset():
    xy, vis, durations, spans = _bank()
    motion = np.array(["constant"] * 4)
    m, err = vf.metrics(xy.copy(), xy, vis, durations, spans, motion)
    assert m["overall"] == m["hidden"] == m["median_px"] == m["rmse_px"] == 0 and m["within5"] == 1
    off = xy.copy()
    off[..., 0] += 6 / 64                                             # 6 px everywhere
    m, err = vf.metrics(off, xy, vis, durations, spans, motion)
    for k in ("overall", "visible", "partial", "hidden", "median_px", "rmse_px"):
        assert abs(m[k] - 6.0) < 1e-4, k
    assert m["within5"] == 0
    assert m["by_duration"][8]["last_hidden_px"] - 6.0 < 1e-4
    assert m["recovery"]["5px"]["all_occluded"]["never_frac"] == 1.0


def test_analysis_never_touches_the_test_bank():
    src = inspect.getsource(vf).replace(vf.__doc__, "")      # code only (the docstring names seed 300)
    assert not re.search(r"SPLITS\[\s*[\"']test", src)
    assert not re.search(r"iter_split\(\s*[\"']test", src)
    assert not re.search(r"\b300\b", src)
    assert (vf.VAL_N, vf.VAL_SEED) == (1000, 200)
