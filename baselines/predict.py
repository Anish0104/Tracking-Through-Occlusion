"""Write a baseline's prediction file for a fixed test bank.

Example (starter test bank: data seed 300, 1,000 videos):
    python -m baselines.predict --model last_pos
    -> preds/last_pos_seed0.npz, data_tag "starter_test_s300"
    python -m baselines.predict --model kalman
    -> preds/kalman_seed0.npz
    python -m baselines.predict --model gru --seed 1 --device mps
    -> loads checkpoints/gru_seed1.pt, writes preds/gru_seed1.npz (n_params 1,067,064)

Only `frames` reaches the model; labels are never passed to it.

Data provenance
    data_tag defaults to "<generator module name>_test_s<data seed>", e.g.
    "starter_test_s300", whichever alias (--data generator or --data data.generator)
    was typed. Exception: for the complete official test bank of data/generator.py
    (--data generator, --n 1000, --data-seed 300; durations 0, 2, 4, 8, 12) the default
    is data.generator.OFFICIAL_TEST_TAG ("generator_test_s300_v1"), the only tag
    evaluation accepts; that tag is refused for any other n or seed.
    The GRU checkpoint and the Kalman config record which generator (module + source
    fingerprint) they were trained / tuned on; predicting on a different generator
    raises unless --allow-data-mismatch.
"""

import argparse
from pathlib import Path

from common.io import save_preds

TEST_SEED = 300      # CONTRACT.md section 4
TEST_SIZE = 1000


def data_identity(data):
    """(module name, source fingerprint) of a data source alias or module path."""
    from baselines import data_cache
    module = data_cache.resolve_module(data)
    return module.__name__, data_cache.source_fingerprint(module)


def official_test_tag(data, data_seed, n):
    """The generator's OFFICIAL_TEST_TAG if (n, data_seed) is its complete official test
    bank (data.generator: 1,000 videos, seed 300, durations 0-12), else None."""
    from baselines import data_cache
    module = data_cache.resolve_module(data)
    tag = getattr(module, "OFFICIAL_TEST_TAG", None)
    test = getattr(module, "SPLITS", {}).get("test")
    if tag and test and (n, data_seed) == (test[0], test[1]):
        return tag
    return None


def default_data_tag(data, data_seed, n=TEST_SIZE):
    """Same tag for every alias of the same generator and seed. For the complete
    official test bank this is the generator's OFFICIAL_TEST_TAG, which is the only
    tag evaluation (data.generator.iter_test_predictions) accepts."""
    official = official_test_tag(data, data_seed, n)
    if official:
        return official
    module, _ = data_identity(data)
    return f"{module.rsplit('.', 1)[-1]}_test_s{data_seed}"


def check_data_tag(data_tag, data, data_seed, n):
    """The official tag is reserved for predictions on the complete, unmodified test bank."""
    from baselines import data_cache
    reserved = getattr(data_cache.resolve_module(data), "OFFICIAL_TEST_TAG", None)
    if reserved and data_tag == reserved and official_test_tag(data, data_seed, n) != reserved:
        raise ValueError(f"data_tag {reserved!r} is reserved for the complete official test bank; "
                         f"got n={n}, data seed {data_seed}")


def check_provenance(what, trained_module, trained_fp, data, allow=False):
    """Raise if `what` was trained/tuned on another generator than `data`."""
    if data is None:
        return
    module, fp = data_identity(data)
    problems = []
    if trained_module is not None and trained_module != module:
        problems.append(f"it was fitted on {trained_module}, but predictions are for {module}")
    elif trained_fp is not None and trained_fp != fp:
        problems.append(f"{module} has changed since it was fitted (source fingerprint differs)")
    if problems and not allow:
        raise ValueError(f"{what}: {problems[0]}. Retrain/retune, or pass --allow-data-mismatch.")


def _kalman_provenance():
    import json
    from baselines.kalman import CONFIG_PATH
    sel = json.loads(CONFIG_PATH.read_text()).get("selection", {}) if CONFIG_PATH.exists() else {}
    module = sel.get("module")
    if module is None and "data" in sel:              # older files: "data.starter.make_batch"
        module = sel["data"].rsplit(".", 1)[0]
    return module, sel.get("fingerprint")


def _predictor(model, seed=0, checkpoint=None, device="cpu", data=None, allow_data_mismatch=False):
    """Return (predict(frames) -> pred_xy, n_params or None).
    With `data`, also check the model was fitted on that generator."""
    if model == "gru":
        from baselines.gru import predict
        from baselines.train_gru import load_checkpoint, resolve_device
        dev = resolve_device(device)
        net, payload = load_checkpoint(checkpoint or f"checkpoints/gru_seed{seed}.pt", dev)
        if payload["config"]["seed"] != seed:
            raise ValueError(f"checkpoint was trained with seed {payload['config']['seed']}, "
                             f"not --seed {seed}")
        cfg = payload["config"]
        trained = cfg.get("data_module") or data_identity(cfg["data"])[0]
        check_provenance(f"GRU checkpoint {checkpoint or f'checkpoints/gru_seed{seed}.pt'}",
                         trained, cfg.get("data_fingerprint"), data, allow_data_mismatch)
        return (lambda frames: predict(frames, net, device=dev)), payload["n_params"]
    if model == "last_pos":
        from baselines.last_pos import predict
        return predict, None
    if model == "kalman":
        from baselines.kalman import predict   # uses baselines/kalman_config.json
        check_provenance("baselines/kalman_config.json", *_kalman_provenance(), data,
                         allow_data_mismatch)
        return predict, None
    raise ValueError(f"unknown or not yet implemented model: {model}")


def run(model="last_pos", data="starter", n=TEST_SIZE, data_seed=TEST_SEED,
        seed=0, out_dir="preds", data_tag=None, checkpoint=None, device="cpu",
        allow_data_mismatch=False):
    from baselines.train_gru import get_make_batch   # same data sources as training

    data_tag = data_tag or default_data_tag(data, data_seed, n)
    check_data_tag(data_tag, data, data_seed, n)      # before any data is generated
    predict, n_params = _predictor(model, seed, checkpoint, device, data, allow_data_mismatch)
    frames = get_make_batch(data)(n=n, seed=data_seed)["frames"]   # pixels only; labels dropped
    pred_xy = predict(frames)

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
    p.add_argument("--data-tag", default=None, help="default: <generator module>_test_s<data-seed>")
    p.add_argument("--checkpoint", default=None, help="gru only; default checkpoints/gru_seed<seed>.pt")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"], help="gru only")
    p.add_argument("--allow-data-mismatch", action="store_true",
                   help="predict even if the model was fitted on a different generator")
    a = p.parse_args(argv)
    path = run(a.model, a.data, a.n, a.data_seed, a.seed, a.out_dir, a.data_tag,
               a.checkpoint, a.device, a.allow_data_mismatch)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
