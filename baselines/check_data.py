"""Check a data source against CONTRACT.md before training on it (seeds 100 and 200 only).

    python -m baselines.check_data --data generator             # Krisha's generator, once it exists
    python -m baselines.check_data --data starter --n 200       # reference run

Checks (exit code 1 if any hard check fails):
    interface   make_batch(n=..., seed=...) works and returns frames / xy / visible_frac
    shapes      frames [N, T, 64, 64], xy [N, T, 2], visible_frac [N, T]; T == 40
    types       float32 everywhere; frames, xy, visible_frac within [0, 1]
    labels      every video has fully visible frames; hidden runs are contiguous
    durations   fully hidden run lengths are within the training set {0, 2, 4, 8}
    seeds       same seed -> identical arrays; train (100) and val (200) share no trajectory
    cache       the baselines cache reproduces make_batch bit for bit (incremental or fallback)
    order       informational: whether make_batch(k, s) is the first k videos of make_batch(n, s)
The test bank (seed 300) is never generated.
"""

import argparse
import sys
import tempfile

import numpy as np

from baselines import data_cache
from baselines.data_cache import DATA_KEYS, TRAIN_SEED, VAL_SEED

T_CONTRACT = 40
TRAIN_DURATIONS = {0, 2, 4, 8}           # CONTRACT.md section 4 (12 is test-only)


def hidden_runs(vis_row):
    """Lengths of maximal runs of fully hidden frames in one video."""
    runs, cur = [], 0
    for h in (np.asarray(vis_row) == 0):
        if h:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    return runs + ([cur] if cur else [])


def check(data, n=200, durations=TRAIN_DURATIONS):
    """Return a list of (status, name, detail) with status in PASS / FAIL / INFO."""
    out = []

    def add(ok, name, detail, info=False):
        out.append(("INFO" if info else "PASS" if ok else "FAIL", name, detail))

    module = data_cache.resolve_module(data)
    splits = {}
    for seed in (TRAIN_SEED, VAL_SEED):
        b = module.make_batch(n=n, seed=seed)
        missing = [k for k in DATA_KEYS if k not in b]
        add(not missing, f"interface s{seed}", f"keys {sorted(b)}" + (f", missing {missing}" if missing else ""))
        if missing:
            return out
        splits[seed] = b
    b = splits[TRAIN_SEED]
    f, xy, vis = (np.asarray(b[k]) for k in DATA_KEYS)
    N, T = f.shape[:2]
    add(f.shape == (n, T, 64, 64) and xy.shape == (n, T, 2) and vis.shape == (n, T), "shapes",
        f"frames {f.shape}, xy {xy.shape}, visible_frac {vis.shape}")
    add(T == T_CONTRACT, "frame count", f"T = {T} (contract default {T_CONTRACT})")
    add(all(a.dtype == np.float32 for a in (f, xy, vis)), "dtypes", f"{f.dtype}, {xy.dtype}, {vis.dtype}")
    add(all(a.min() >= 0 and a.max() <= 1 for a in (f, xy, vis)), "value ranges",
        f"frames [{f.min():.3g}, {f.max():.3g}], xy [{xy.min():.3g}, {xy.max():.3g}], "
        f"vis [{vis.min():.3g}, {vis.max():.3g}]")
    add(True, "frame values", f"{len(np.unique(f))} distinct (uint8 cache needs <= 256)", info=True)
    add(bool((vis == 1).any(axis=1).all()), "visible frames", "every video has fully visible frames")
    runs = [hidden_runs(v) for v in vis]
    lengths = sorted({r for rs in runs for r in rs} | ({0} if any(not rs for rs in runs) else set()))
    add(all(len(rs) <= 1 for rs in runs), "single occlusion",
        f"{sum(len(rs) > 1 for rs in runs)} videos with more than one hidden run")
    add(set(lengths) <= set(durations), "hidden durations",
        f"observed {lengths}, allowed {sorted(durations)}")
    again = module.make_batch(n=n, seed=TRAIN_SEED)
    add(all(data_cache._bitwise_equal(again[k], b[k]) for k in DATA_KEYS), "seed determinism",
        f"make_batch(n={n}, seed={TRAIN_SEED}) twice")
    shared = {v.tobytes() for v in xy} & {v.tobytes() for v in np.asarray(splits[VAL_SEED]["xy"])}
    add(not shared, "split disjointness", f"{len(shared)} identical trajectories in train and val samples")
    k = min(8, n)
    prefix = all(data_cache._bitwise_equal(np.asarray(module.make_batch(n=k, seed=TRAIN_SEED)[key]),
                                           np.asarray(b[key])[:k]) for key in DATA_KEYS)
    add(True, "prefix order", f"make_batch({k}) {'is' if prefix else 'is NOT'} the first {k} of "
        f"make_batch({n}); fine either way, the cache always reproduces make_batch(n) itself", info=True)
    with tempfile.TemporaryDirectory() as tmp:
        cached = data_cache.get_split(data, n, TRAIN_SEED, tmp, log=lambda *a: None)
        producer = data_cache.incremental_producer(module)[0] or "make_batch (fallback)"
        add(all(data_cache._bitwise_equal(np.asarray(cached[key]), b[key]) for key in DATA_KEYS),
            "cache exactness", f"via {producer}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="generator")
    p.add_argument("--n", type=int, default=200, help="videos per split sample")
    a = p.parse_args(argv)
    results = check(a.data, a.n)
    for status, name, detail in results:
        print(f"{status:4s}  {name:18s} {detail}")
    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\n{len(failed)} failed check(s) for --data {a.data}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
