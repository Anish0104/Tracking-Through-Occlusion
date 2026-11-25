"""Grid-search the Kalman hyperparameters on the validation bank (data seed 200).

    python -m baselines.tune_kalman            # search, print the top configs
    python -m baselines.tune_kalman --write    # also save baselines/kalman_config.json

Objective: overall mean pixel error (per video, then over videos; see
dev_metrics.py) on all frames of the validation bank. Visible / partial /
hidden errors are reported alongside but not optimized directly.
The test bank (seed 300) is never generated here. The detector runs once;
each config only re-runs the filter. Deterministic: same grid, same result.
"""

import argparse
import itertools
import json
import time

from baselines.detector import detect
from baselines.dev_metrics import VAL_SEED, format_table, summarize
from baselines.kalman import CONFIG_PATH, KalmanConfig, config_dict, track

GRID = {
    "accel_px": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    "meas_px": [0.1, 0.2, 0.4, 0.8],
    "partial_px": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
    "min_pixels": [1, 4, 8],
}
FIXED = {"full_px": 28.0, "init_vel_px": 2.0}


def search(frames, xy, vis, grid=GRID, fixed=FIXED):
    det = detect(frames)
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = KalmanConfig(**dict(zip(keys, values)), **fixed)
        res = summarize(track(*det, cfg), xy, vis)
        rows.append((res["overall"]["err_px"], cfg, res))
    rows.sort(key=lambda r: r[0])          # stable sort: ties keep grid order
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000, help="validation videos (contract: 1,000)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--write", action="store_true", help=f"save the best config to {CONFIG_PATH.name}")
    a = p.parse_args(argv)

    from data.starter import make_batch
    b = make_batch(n=a.n, seed=VAL_SEED)
    t0 = time.time()
    rows = search(b["frames"], b["xy"], b["visible_frac"])
    print(f"{len(rows)} configs on starter val seed {VAL_SEED} ({a.n} videos) in {time.time() - t0:.1f}s")
    print(f"{'overall':>8s} {'visible':>8s} {'partial':>8s} {'hidden':>8s}  config")
    for err, cfg, res in rows[:a.top]:
        print(f"{err:8.3f} {res['visible']['err_px']:8.3f} {res['partial']['err_px']:8.3f} "
              f"{res['hidden']['err_px']:8.3f}  accel={cfg.accel_px} meas={cfg.meas_px} "
              f"partial={cfg.partial_px} min_px={cfg.min_pixels}")

    best_err, best, best_res = rows[0]
    print("\nbest config on validation:\n" + format_table({"kalman": best_res}))
    if a.write:
        doc = {
            "_note": "Kalman hyperparameters chosen by baselines/tune_kalman.py. "
                     "Units: pixels and frames. Selected on validation data only.",
            "params": config_dict(best),
            "selection": {
                "data": "data.starter.make_batch",
                "data_seed": VAL_SEED,
                "n_videos": a.n,
                "objective": "overall mean pixel error, per video then over videos",
                "grid": GRID,
                "fixed": FIXED,
                "n_configs": len(rows),
            },
            "validation": {cat: {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                           for cat, r in best_res.items()},
        }
        CONFIG_PATH.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()
