"""Core tracking metrics. Pure numpy, no model code, no torch (CONTRACT.md section 3).

Everything here takes arrays and returns numbers, so it is easy to test and
reuse from report.py, the figures and demo/animate.py.

Conventions shared with the rest of the team (baselines/dev_metrics.py):

* Error is the center error in **pixels**: the normalized difference x SIZE.
* Frame categories come from ``visible_frac``: fully visible (== 1),
  fully hidden (== 0) and partial (strictly between).
* Aggregation is **mean per video, then mean over videos** that contain at
  least one frame of that category (eval/README.md). A plain frame-weighted
  mean would let long videos and common categories dominate.
* Success means error <= ``SUCCESS_PX`` (5 px, CONTRACT.md section 4).

Hidden duration is not stored in the batch dict, so we recover it from the
labels: the number of frames with ``visible_frac == 0``. The generator builds
one contiguous hidden interval per video, so this equals the requested
duration (0, 2, 4, 8 or 12).
"""

import numpy as np

SIZE = 64            # frames are SIZE x SIZE; normalized units x SIZE = pixels
SUCCESS_PX = 5.0     # CONTRACT.md section 4: success threshold
RECOVERY_PX = 5.0    # CONTRACT.md section 4: recovery threshold
RECOVERY_FRAMES = 3  # consecutive good frames needed to count as recovered

#: Frame categories, as boolean masks over ``visible_frac``.
CATEGORIES = ("overall", "visible", "partial", "hidden")


def pixel_error(pred_xy, true_xy):
    """Center error in pixels, per frame. -> [N, T]

    Computed in float64 so that sub-pixel differences between models (the
    comparison that actually matters here) are not lost to float32 rounding.
    """
    pred_xy = np.asarray(pred_xy, np.float64)
    true_xy = np.asarray(true_xy, np.float64)
    if pred_xy.shape != true_xy.shape:
        raise ValueError(f"shape mismatch: pred {pred_xy.shape} vs true {true_xy.shape}")
    if pred_xy.ndim != 3 or pred_xy.shape[-1] != 2:
        raise ValueError("expected [N, T, 2] arrays")
    return np.linalg.norm(pred_xy - true_xy, axis=-1) * SIZE


def category_masks(visible_frac):
    """Boolean frame masks per category. -> {name: [N, T] bool}"""
    v = np.asarray(visible_frac, np.float64)
    return {
        "overall": np.ones(v.shape, bool),
        "visible": v >= 1.0,
        "partial": (v > 0.0) & (v < 1.0),
        "hidden": v <= 0.0,
    }


def hidden_durations(visible_frac):
    """Number of fully hidden frames per video. -> [N] int

    This is the video's hidden duration (the generator uses one contiguous
    hidden interval), which is how results are grouped in the report.
    """
    return (np.asarray(visible_frac, np.float64) <= 0.0).sum(axis=1).astype(int)


def per_video_mean(values, mask):
    """Mean within each video over ``mask``, then over videos that have any.

    Returns ``(mean, n_videos)``; the mean is nan when no video qualifies.
    Videos without a frame of this category are dropped rather than counted
    as zero, which would otherwise bias the average downwards.
    """
    values = np.asarray(values, np.float64)
    mask = np.asarray(mask, bool)
    count = mask.sum(axis=1)
    keep = count > 0
    if not keep.any():
        return float("nan"), 0
    per_video = (values * mask).sum(axis=1)[keep] / count[keep]
    return float(per_video.mean()), int(keep.sum())


def per_video_values(values, mask):
    """Per-video means over ``mask``, nan where a video has no such frame. -> [N]

    Kept separate from :func:`per_video_mean` because the paired comparisons
    and bootstrap intervals need the individual video numbers, not the mean.
    """
    values = np.asarray(values, np.float64)
    mask = np.asarray(mask, bool)
    count = mask.sum(axis=1)
    out = np.full(values.shape[0], np.nan)
    keep = count > 0
    out[keep] = (values * mask).sum(axis=1)[keep] / count[keep]
    return out


def reappearance_index(visible_frac):
    """First fully visible frame after the hidden interval. -> [N] int, -1 if none

    Recovery is measured from the moment the target is fully visible again, so
    partially visible frames at the edge of the barrier do not start the clock.
    """
    v = np.asarray(visible_frac, np.float64)
    n, t = v.shape
    hidden = v <= 0.0
    full = v >= 1.0
    out = np.full(n, -1)
    for i in range(n):
        h = np.flatnonzero(hidden[i])
        if h.size == 0:
            continue
        after = np.flatnonzero(full[i][h[-1] + 1:])
        if after.size:
            out[i] = h[-1] + 1 + after[0]
    return out


def recovery_delay(err_px, visible_frac, threshold=RECOVERY_PX,
                   n_consecutive=RECOVERY_FRAMES):
    """Frames from full reappearance until the error stays small.

    "Small" means ``threshold`` or less for ``n_consecutive`` frames in a row
    (CONTRACT.md section 4). Returns ``(delay, recovered)``, both [N]:

    * ``delay`` is the offset of the first frame of that run, 0 when the model
      is already accurate on the first visible frame, and nan where the video
      has no occlusion or never recovers.
    * ``recovered`` is False when the run never happens (and for videos with
      no occlusion), so the caller can report the never-recovered fraction
      separately from the delay average.

    A video whose tail is too short to hold ``n_consecutive`` frames cannot
    show a qualifying run, so it counts as not recovered.
    """
    err_px = np.asarray(err_px, np.float64)
    start = reappearance_index(visible_frac)
    n = err_px.shape[0]
    delay = np.full(n, np.nan)
    recovered = np.zeros(n, bool)
    for i in range(n):
        s = start[i]
        if s < 0:
            continue
        good = err_px[i, s:] <= threshold
        run = 0
        for k, g in enumerate(good):
            run = run + 1 if g else 0
            if run >= n_consecutive:
                delay[i] = k - n_consecutive + 1
                recovered[i] = True
                break
    return delay, recovered


def has_occlusion(visible_frac):
    """Videos with at least one fully hidden frame. -> [N] bool"""
    return (np.asarray(visible_frac, np.float64) <= 0.0).any(axis=1)


def has_reappearance(visible_frac, n_consecutive=RECOVERY_FRAMES):
    """Videos where recovery is a fair question to ask. -> [N] bool

    True when the video is occluded, becomes fully visible again, **and** has
    at least ``n_consecutive`` frames left after that point.

    The last condition matters: the contract defines recovery as a run of
    ``n_consecutive`` good frames, so a video that ends sooner cannot show one
    even with zero error. Including those videos would put a floor under every
    model's never-recovered rate and make the metric describe the data instead
    of the tracker (on the starter bank, a perfect tracker would still score
    13.6% "never recovered"). They are excluded from the denominator instead.
    """
    idx = reappearance_index(visible_frac)
    t = np.asarray(visible_frac).shape[1]
    return (idx >= 0) & (idx <= t - n_consecutive)


def spikes_per_neuron_per_frame(spikes, n_neurons):
    """Mean spikes per neuron per frame, the SNN's activity/energy proxy.

    ``spikes`` is [N, T] total spikes per frame (CONTRACT.md section 3).
    """
    if spikes is None or not n_neurons:
        return float("nan")
    return float(np.asarray(spikes, np.float64).mean() / n_neurons)


def summarize(pred_xy, true_xy, visible_frac, spikes=None, n_neurons=None):
    """All single-model metrics for one prediction set.

    Returns a nested dict: per-category error/success, recovery statistics and
    optional spike activity. Used for both the printed tables and the figures.
    """
    err = pixel_error(pred_xy, true_xy)
    masks = category_masks(visible_frac)
    out = {"n_videos": int(err.shape[0]), "n_frames": int(err.size), "categories": {}}
    for name in CATEGORIES:
        m = masks[name]
        err_mean, n_vid = per_video_mean(err, m)
        success, _ = per_video_mean((err <= SUCCESS_PX).astype(float), m)
        out["categories"][name] = {
            "err_px": err_mean,
            "success": success,
            "frames": int(m.sum()),
            "videos": n_vid,
        }

    occ = has_occlusion(visible_frac)
    reap = has_reappearance(visible_frac)
    delay, recovered = recovery_delay(err, visible_frac)
    # Report delay over recovered videos only; never-recovered videos have no
    # finite delay and are counted by their own fraction instead. The
    # denominator is videos that actually reappear (see has_reappearance).
    out["recovery"] = {
        "videos_with_occlusion": int(occ.sum()),
        "videos_with_reappearance": int(reap.sum()),
        "delay_mean": float(np.nanmean(delay[recovered])) if recovered.any() else float("nan"),
        "delay_median": float(np.nanmedian(delay[recovered])) if recovered.any() else float("nan"),
        "never_recovered": float((~recovered)[reap].mean()) if reap.any() else float("nan"),
        "recovered": int(recovered.sum()),
    }
    if spikes is not None and n_neurons:
        out["spikes_per_neuron_per_frame"] = spikes_per_neuron_per_frame(spikes, n_neurons)
        out["spikes_per_frame"] = float(np.asarray(spikes, np.float64).mean())
    return out


def summarize_by_duration(pred_xy, true_xy, visible_frac, spikes=None, n_neurons=None):
    """:func:`summarize` for each hidden duration present. -> {duration: summary}"""
    dur = hidden_durations(visible_frac)
    out = {}
    for d in sorted(set(dur.tolist())):
        sel = dur == d
        out[int(d)] = summarize(
            np.asarray(pred_xy)[sel], np.asarray(true_xy)[sel], np.asarray(visible_frac)[sel],
            spikes=None if spikes is None else np.asarray(spikes)[sel], n_neurons=n_neurons)
    return out
