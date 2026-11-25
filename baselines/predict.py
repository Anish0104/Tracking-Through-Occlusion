"""Write a baseline's prediction file for a fixed test bank.

Example (starter test bank: data seed 300, 1,000 videos):
    python -m baselines.predict --model last_pos
    -> preds/last_pos_seed0.npz, data_tag "starter_test_s300"
    python -m baselines.predict --model kalman
    -> preds/kalman_seed0.npz
    python -m baselines.predict --model gru --seed 1 --device mps
    -> loads checkpoints/gru_seed1.pt, writes preds/gru_seed1.npz (n_params 1,067,064)

Only `frames` reaches the model; labels are never passed to it.
"""

import argparse
from pathlib import Path

from common.io import save_preds

TEST_SEED = 300      # CONTRACT.md section 4
TEST_SIZE = 1000


def _predictor(model, seed=0, checkpoint=None, device="cpu"):
    """Return (predict(frames) -> pred_xy, n_params or None)."""
    if model == "gru":
        from baselines.gru import predict
        from baselines.train_gru import load_checkpoint, resolve_device
        dev = resolve_device(device)
        net, payload = load_checkpoint(checkpoint or f"checkpoints/gru_seed{seed}.pt", dev)
        if payload["config"]["seed"] != seed:
            raise ValueError(f"checkpoint was trained with seed {payload['config']['seed']}, "
                             f"not --seed {seed}")
        return (lambda frames: predict(frames, net, device=dev)), payload["n_params"]
    if model == "last_pos":
        from baselines.last_pos import predict
        return predict, None
    if model == "kalman":
        from baselines.kalman import predict   # uses baselines/kalman_config.json
        return predict, None
    raise ValueError(f"unknown or not yet implemented model: {model}")


def run(model="last_pos", data="starter", n=TEST_SIZE, data_seed=TEST_SEED,
        seed=0, out_dir="preds", data_tag=None, checkpoint=None, device="cpu"):
    from baselines.train_gru import get_make_batch   # same data sources as training

    predict, n_params = _predictor(model, seed, checkpoint, device)
    frames = get_make_batch(data)(n=n, seed=data_seed)["frames"]   # pixels only; labels dropped
    pred_xy = predict(frames)

    data_tag = data_tag or f"{data}_test_s{data_seed}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}_seed{seed}.npz"
    save_preds(path, pred_xy, model=model, seed=seed, data_tag=data_tag, n_params=n_params)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="last_pos", choices=["last_pos", "kalman", "gru"])
    p.add_argument("--data", default="starter",
                   help="data source: starter, generator, or a module path with make_batch(n, seed)")
    p.add_argument("--n", type=int, default=TEST_SIZE)
    p.add_argument("--data-seed", type=int, default=TEST_SEED)
    p.add_argument("--seed", type=int, default=0, help="training seed (0 for non-learned baselines)")
    p.add_argument("--out-dir", default="preds")
    p.add_argument("--data-tag", default=None, help="default: <data>_test_s<data-seed>")
    p.add_argument("--checkpoint", default=None, help="gru only; default checkpoints/gru_seed<seed>.pt")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"], help="gru only")
    a = p.parse_args(argv)
    path = run(a.model, a.data, a.n, a.data_seed, a.seed, a.out_dir, a.data_tag,
               a.checkpoint, a.device)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
