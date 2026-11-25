"""Write a baseline's prediction file for a fixed test bank.

Example (starter test bank: data seed 300, 1,000 videos):
    python -m baselines.predict --model last_pos
    -> preds/last_pos_seed0.npz, data_tag "starter_test_s300"
    python -m baselines.predict --model kalman
    -> preds/kalman_seed0.npz

Only `frames` reaches the model; labels are never passed to it.
"""

import argparse
from pathlib import Path

from common.io import save_preds

TEST_SEED = 300      # CONTRACT.md section 4
TEST_SIZE = 1000


def _predictor(model):
    if model == "last_pos":
        from baselines.last_pos import predict
        return predict
    if model == "kalman":
        from baselines.kalman import predict   # uses baselines/kalman_config.json
        return predict
    raise ValueError(f"unknown or not yet implemented model: {model}")


def run(model="last_pos", data="starter", n=TEST_SIZE, data_seed=TEST_SEED,
        seed=0, out_dir="preds", data_tag=None):
    if data != "starter":
        raise ValueError("only the starter generator is available so far")
    from data.starter import make_batch

    predict = _predictor(model)
    frames = make_batch(n=n, seed=data_seed)["frames"]   # pixels only; labels dropped here
    pred_xy = predict(frames)

    data_tag = data_tag or f"{data}_test_s{data_seed}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}_seed{seed}.npz"
    save_preds(path, pred_xy, model=model, seed=seed, data_tag=data_tag)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="last_pos", choices=["last_pos", "kalman"])
    p.add_argument("--data", default="starter", choices=["starter"])
    p.add_argument("--n", type=int, default=TEST_SIZE)
    p.add_argument("--data-seed", type=int, default=TEST_SEED)
    p.add_argument("--seed", type=int, default=0, help="training seed (0 for non-learned baselines)")
    p.add_argument("--out-dir", default="preds")
    p.add_argument("--data-tag", default=None, help="default: <data>_test_s<data-seed>")
    a = p.parse_args(argv)
    path = run(a.model, a.data, a.n, a.data_seed, a.seed, a.out_dir, a.data_tag)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
