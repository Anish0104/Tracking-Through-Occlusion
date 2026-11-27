"""Statistics: bootstrap confidence intervals and paired model comparisons.

Why paired: every model is run on the *same* test videos, so a video's
difficulty is common to all of them. Comparing per-video differences removes
that shared variation, which is what makes sub-pixel differences between
models detectable at all. The baselines handoff reports that the 5 px success
and recovery metrics saturate, so these paired intervals are the evidence that
separates the models.

The bootstrap resamples **videos** (not frames), because frames inside one
video are strongly dependent and resampling them would understate the
interval. No scipy: numpy only, so eval/ has no extra dependency.
"""

import numpy as np

N_BOOT = 10000     # bootstrap resamples; enough for a stable 95% interval
CI_LEVEL = 0.95
SEED = 12345       # fixed so reported intervals are reproducible


def _clean(x):
    x = np.asarray(x, np.float64)
    return x[np.isfinite(x)]


def bootstrap_mean_ci(values, n_boot=N_BOOT, level=CI_LEVEL, seed=SEED):
    """Percentile bootstrap CI for the mean of per-video values.

    Returns ``(mean, low, high, n)``; nan bounds when there is nothing to
    resample. Non-finite entries (videos without the relevant frames) are
    dropped first.
    """
    v = _clean(values)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(v.mean())
    if n == 1:
        return mean, mean, mean, 1
    rng = np.random.default_rng(seed)
    # One [n_boot, n] draw of indices: vectorized, so 10k resamples stay fast.
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    half = (1.0 - level) / 2.0
    low, high = np.quantile(means, [half, 1.0 - half])
    return mean, float(low), float(high), n


def paired_diff_ci(a, b, n_boot=N_BOOT, level=CI_LEVEL, seed=SEED):
    """Bootstrap CI for the mean paired difference ``a - b``.

    ``a`` and ``b`` are per-video values for two models on the same videos, in
    the same order. Videos where either side is non-finite are dropped so the
    pairing stays intact. The interval excluding 0 is what we read as "the
    difference is resolved at this sample size".

    Returns a dict with the mean difference, interval, the fraction of videos
    where ``a`` is lower (better, for errors), and the pair count.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same videos on both sides")
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    n = d.size
    if n == 0:
        return {"mean_diff": float("nan"), "low": float("nan"), "high": float("nan"),
                "n_pairs": 0, "frac_a_better": float("nan"), "significant": False}
    mean = float(d.mean())
    if n == 1:
        low = high = mean
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = d[idx].mean(axis=1)
        half = (1.0 - level) / 2.0
        lo, hi = np.quantile(boot, [half, 1.0 - half])
        low, high = float(lo), float(hi)
    return {
        "mean_diff": mean,
        "low": low,
        "high": high,
        "n_pairs": n,
        "frac_a_better": float((d < 0).mean()),
        "significant": bool(low > 0 or high < 0),
    }


def mean_sd(values):
    """Mean and sample spread across seeds. -> (mean, sd, n)

    Uses ddof=1 (sample sd) because the three training seeds are a sample of
    the seed-to-seed variation, not the whole population. The handoff notes
    this spread is larger than one seed's sampling CI, so it must be reported.
    """
    v = _clean(values)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    if v.size == 1:
        return float(v[0]), 0.0, 1
    return float(v.mean()), float(v.std(ddof=1)), int(v.size)
