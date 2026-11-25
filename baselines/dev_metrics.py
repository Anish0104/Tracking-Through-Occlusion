"""Development metrics for tuning and sanity checks inside baselines/.

This is NOT the project's evaluation (that is eval/, owned by Kunal). It only
helps choose baseline hyperparameters on the validation bank. Labels are used
here to score predictions, never to make them.

Aggregation: for each frame category, average the per-frame pixel error within
each video, then average over the videos that have at least one such frame
(eval/README.md: "mean per video, then over videos"). Success = error <= 5 px.

    python -m baselines.dev_metrics                    # last_pos vs kalman on val seed 200
"""

import argparse

import numpy as np

SIZE = 64
VAL_SEED = 200
TEST_SEED = 300
SUCCESS_PX = 5.0

CATEGORIES = {
    "overall": lambda v: np.ones_like(v, bool),
    "visible": lambda v: v == 1,
    "partial": lambda v: (v > 0) & (v < 1),
    "hidden": lambda v: v == 0,
}


def pixel_error(pred_xy, xy):
    return np.linalg.norm(np.asarray(pred_xy, np.float64) - xy, axis=-1) * SIZE   # [N, T]


def _per_video_mean(values, mask):
    cnt = mask.sum(axis=1)
    keep = cnt > 0
    if not keep.any():
        return float("nan")
    return float(((values * mask).sum(axis=1)[keep] / cnt[keep]).mean())


def summarize(pred_xy, xy, visible_frac):
    """Return {category: {"err_px", "success", "frames", "videos"}}."""
    err = pixel_error(pred_xy, xy)
    out = {}
    for name, sel in CATEGORIES.items():
        m = sel(visible_frac)
        out[name] = {
            "err_px": _per_video_mean(err, m),
            "success": _per_video_mean((err <= SUCCESS_PX).astype(float), m),
            "frames": int(m.sum()),
            "videos": int((m.sum(axis=1) > 0).sum()),
        }
    return out


def format_table(results):
    """results: {model: summarize(...)} -> printable table."""
    lines = [f"{'model':10s} {'category':9s} {'frames':>7s} {'videos':>6s} {'mean err px':>12s} {'<=5px':>7s}"]
    for model, res in results.items():
        for cat, r in res.items():
            lines.append(f"{model:10s} {cat:9s} {r['frames']:7d} {r['videos']:6d} "
                         f"{r['err_px']:12.3f} {r['success']:7.3f}")
    return "\n".join(lines)


def check_not_test(data_seed, allow_test=False):
    if data_seed == TEST_SEED and not allow_test:
        raise SystemExit("refusing to score the held-out test bank (seed 300) during development; "
                         "pass --allow-test only for final reporting")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["last_pos", "kalman"], choices=["last_pos", "kalman"])
    p.add_argument("--data-seed", type=int, default=VAL_SEED)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--allow-test", action="store_true")
    a = p.parse_args(argv)
    check_not_test(a.data_seed, a.allow_test)

    from baselines.predict import _predictor
    from data.starter import make_batch

    b = make_batch(n=a.n, seed=a.data_seed)
    results = {m: summarize(_predictor(m)(b["frames"]), b["xy"], b["visible_frac"]) for m in a.models}
    print(f"starter data seed {a.data_seed}, {a.n} videos")
    print(format_table(results))


if __name__ == "__main__":
    main()
