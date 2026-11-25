"""Train the CNN-GRU baseline (one model-initialization seed per run).

Fixed by CONTRACT.md section 4:
    loss          common/loss.py: balanced_mse
    optimizer     Adam, lr 1e-3, batch 16, at most 30 epochs
    data          train = make_batch(5000, seed=100), validation = make_batch(1000, seed=200)
    model seeds   0, 1, 2 (each run trains one of them, from scratch)
Early stopping watches validation balanced MSE (patience 5 provisionally, pending
agreement with Anish so both neural models share the same rule). The checkpoint
with the lowest validation loss is saved to checkpoints/gru_seed<k>.pt.

The model only ever sees `frames`; `xy` and `visible_frac` are used by the loss only.
The test bank (data seed 300) is never touched here.

Examples:
    python -m baselines.train_gru --seed 0 --device mps
    python -m baselines.data_cache --data starter --cache-dir checkpoints/data_cache   # build once
    python -m baselines.train_gru --seed 1 --device mps --cache-dir checkpoints/data_cache
    python -m baselines.train_gru --smoke --device cpu          # tiny run, ~1 minute
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from baselines import data_cache
from baselines.data_cache import DATA_SOURCES  # noqa: F401  (re-exported)
from baselines.gru import EXPECTED_PARAMS, build_model, count_params
from common.loss import balanced_mse

# CONTRACT.md section 4
TRAIN_SEED, VAL_SEED = 100, 200
TRAIN_SIZE, VAL_SIZE = 5000, 1000
LR, BATCH_SIZE, MAX_EPOCHS = 1e-3, 16, 30
PATIENCE = 5                       # provisional, pending agreement with Anish

SMOKE = dict(n_train=48, n_val=16, epochs=2)
DATA_KEYS = ("frames", "xy", "visible_frac")


# ---------------------------------------------------------------- data

# Each data source is a module exposing make_batch(n, seed) -> dict (CONTRACT.md section 1).
# "generator" is Krisha's final generator; it is imported lazily, so it only has to
# exist when it is selected. Any other "package.module" path also works.

def get_make_batch(source):
    """Resolve a data source name (or a dotted module path) to its make_batch function."""
    return data_cache.resolve_module(source).make_batch


def load_split(source, n, seed, cache_dir=None, cache_format="auto"):
    """Return {frames, xy, visible_frac} for make_batch(n, seed).

    Without cache_dir: in-memory numpy arrays straight from make_batch.
    With cache_dir: the exact same values from baselines/data_cache.py (built once,
    incrementally where possible, then memory-mapped read-only and shared by all
    model seeds). Only the current batch is ever copied out and sent to the device.
    """
    if cache_dir is None:
        batch = get_make_batch(source)(n=n, seed=seed)
        return {k: batch[k] for k in DATA_KEYS}
    return data_cache.get_split(source, n, seed, cache_dir, fmt=cache_format)


def check_split(split):
    n, t = split["frames"].shape[:2]
    assert split["frames"].shape[2:] == (64, 64), split["frames"].shape
    assert split["xy"].shape == (n, t, 2), split["xy"].shape
    assert split["visible_frac"].shape == (n, t), split["visible_frac"].shape


def _take(arr, idx):
    """arr[idx] with idx's order preserved. Disk-backed arrays are read in sorted
    order (sequential I/O) and then put back in idx order, so a batch is bit-identical
    whether the split lives in RAM or in the cache."""
    if type(arr) is np.ndarray:
        return arr[idx]
    perm = np.argsort(idx, kind="stable")
    out = np.asarray(arr[idx[perm]])
    return out[np.argsort(perm, kind="stable")]


def iterate_batches(split, batch_size, order=None):
    """Yield torch batches (frames, xy, visible_frac) in `order` (default: sequential).
    Only one batch is copied out of the split at a time."""
    n = len(split["frames"])
    order = np.arange(n) if order is None else np.asarray(order)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        yield tuple(torch.from_numpy(np.ascontiguousarray(_take(split[k], idx), np.float32))
                    for k in DATA_KEYS)


# ---------------------------------------------------------------- reproducibility, device

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                 # also seeds CUDA and MPS
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name):
    """'cpu', 'cuda', 'mps' or 'auto' (cuda > mps > cpu). Fails loudly if unavailable."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else \
            "mps" if torch.backends.mps.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    return torch.device(name)


# ---------------------------------------------------------------- early stopping

class EarlyStopping:
    """Stop after `patience` epochs without the monitored loss improving by more than
    `min_delta`. patience=None disables stopping (train for the full epoch cap)."""

    def __init__(self, patience=PATIENCE, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = math.inf
        self.best_epoch = None
        self.bad_epochs = 0

    def update(self, loss, epoch):
        """Record one epoch. Returns True when `loss` is a new best."""
        if loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = loss, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self):
        return self.patience is not None and self.bad_epochs >= self.patience


# ---------------------------------------------------------------- training

@dataclass
class TrainConfig:
    seed: int = 0
    data: str = "starter"
    n_train: int = TRAIN_SIZE
    n_val: int = VAL_SIZE
    train_seed: int = TRAIN_SEED
    val_seed: int = VAL_SEED
    epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    patience: int = PATIENCE         # <= 0 disables early stopping
    min_delta: float = 0.0
    device: str = "auto"
    checkpoint: str = ""             # default: checkpoints/gru_seed<seed>.pt
    cache_dir: str = ""
    cache_format: str = "auto"       # auto | uint8 | raw (see baselines/data_cache.py)
    smoke: bool = False
    log_every: int = 50

    def __post_init__(self):
        if self.train_seed == self.val_seed:
            raise ValueError("training and validation data seeds must differ")
        if 300 in (self.train_seed, self.val_seed):
            raise ValueError("data seed 300 is the test bank; never train or validate on it")
        if not self.checkpoint:
            self.checkpoint = f"checkpoints/gru_seed{self.seed}.pt"


def train_one_epoch(model, split, optimizer, batch_size, device, rng, log_every=0):
    """One pass over the training split in a seeded random order.
    Returns the mean balanced MSE over training sequences."""
    model.train()
    order = rng.permutation(len(split["frames"]))
    total, count = 0.0, 0
    for step, (frames, xy, vis) in enumerate(iterate_batches(split, batch_size, order), 1):
        frames, xy, vis = frames.to(device), xy.to(device), vis.to(device)
        pred, _ = model(frames)                 # frames only
        loss = balanced_mse(pred, xy, vis)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(frames)
        count += len(frames)
        if log_every and step % log_every == 0:
            print(f"    step {step:4d}  train loss {total / count:.6f}", flush=True)
    return total / count


@torch.no_grad()
def evaluate(model, split, batch_size, device):
    """Mean balanced MSE over all sequences of a split (batch means weighted by size,
    so the result does not depend on the batch size)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for frames, xy, vis in iterate_batches(split, batch_size):
        pred, _ = model(frames.to(device))      # frames only
        total += balanced_mse(pred, xy.to(device), vis.to(device)).item() * len(frames)
        count += len(frames)
    model.train(was_training)
    return total / count


def save_checkpoint(path, model, optimizer, epoch, val_loss, history, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "gru",
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "history": history,
        "config": dict(config),
        "n_params": int(count_params(model)),
        "torch_version": str(torch.__version__),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)                           # atomic: a crash never corrupts the best file


def load_checkpoint(path, device="cpu"):
    """Rebuild the CNN-GRU from a checkpoint. Returns (model in eval mode, payload)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model") != "gru":
        raise ValueError(f"{path} is not a GRU checkpoint")
    model = build_model(seed=0)                 # weights are overwritten below
    model.load_state_dict(payload["model_state"])
    n = count_params(model)
    if n != payload["n_params"] or n != EXPECTED_PARAMS:
        raise ValueError(f"parameter count mismatch: model {n}, checkpoint {payload['n_params']}")
    return model.to(device).eval(), payload


def train(cfg: TrainConfig, train_split=None, val_split=None):
    """Train one seed. Splits may be passed in (tests); otherwise they are generated.
    Returns a summary dict (also written next to the checkpoint as JSON)."""
    device = resolve_device(cfg.device)
    cache = cfg.cache_dir or None
    if train_split is None:
        train_split = load_split(cfg.data, cfg.n_train, cfg.train_seed, cache, cfg.cache_format)
    if val_split is None:
        val_split = load_split(cfg.data, cfg.n_val, cfg.val_seed, cache, cfg.cache_format)
    check_split(train_split)
    check_split(val_split)

    seed_everything(cfg.seed)
    model = build_model(seed=cfg.seed).to(device)
    n_params = count_params(model)
    assert n_params == EXPECTED_PARAMS, n_params
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    shuffle_rng = np.random.default_rng(cfg.seed)   # batch order depends on the model seed only
    stopper = EarlyStopping(cfg.patience if cfg.patience > 0 else None, cfg.min_delta)

    print(f"CNN-GRU seed {cfg.seed} | {n_params:,} params | device {device} | "
          f"train {len(train_split['frames'])} (data seed {cfg.train_seed}) | "
          f"val {len(val_split['frames'])} (data seed {cfg.val_seed})", flush=True)

    history = []
    stopped_early = False
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_split, optimizer, cfg.batch_size, device,
                                     shuffle_rng, cfg.log_every)
        val_loss = evaluate(model, val_split, cfg.batch_size, device)
        improved = stopper.update(val_loss, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": round(time.time() - t0, 2)})
        if improved:
            save_checkpoint(cfg.checkpoint, model, optimizer, epoch, val_loss, history, asdict(cfg))
        print(f"epoch {epoch:2d}  train {train_loss:.6f}  val {val_loss:.6f}"
              f"{'  *best, saved' if improved else ''}  ({history[-1]['seconds']}s)", flush=True)
        if stopper.should_stop:
            stopped_early = True
            print(f"early stop: no val improvement for {stopper.patience} epochs", flush=True)
            break

    summary = {"seed": cfg.seed, "checkpoint": cfg.checkpoint, "best_epoch": stopper.best_epoch,
               "best_val_loss": stopper.best, "epochs_run": len(history),
               "stopped_early": stopped_early, "n_params": n_params,
               "data_fingerprint": data_cache.source_fingerprint(data_cache.resolve_module(cfg.data)),
               "history": history, "config": asdict(cfg)}
    Path(cfg.checkpoint).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"best val {stopper.best:.6f} at epoch {stopper.best_epoch} -> {cfg.checkpoint}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="model initialization seed (0, 1 or 2)")
    p.add_argument("--data", default="starter",
                   help="data source: starter, generator, or a module path with make_batch(n, seed)")
    p.add_argument("--n-train", type=int, default=None, help=f"default {TRAIN_SIZE}")
    p.add_argument("--n-val", type=int, default=None, help=f"default {VAL_SIZE}")
    p.add_argument("--epochs", type=int, default=None, help=f"maximum epochs, default {MAX_EPOCHS}")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--patience", type=int, default=PATIENCE, help="early-stopping patience; 0 disables")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--checkpoint", default="", help="default: checkpoints/gru_seed<seed>.pt")
    p.add_argument("--cache-dir", default="",
                   help="shared on-disk cache for the splits (see baselines/data_cache.py); "
                        "default: generate in memory")
    p.add_argument("--cache-format", default="auto", choices=data_cache.FORMATS)
    p.add_argument("--smoke", action="store_true",
                   help=f"tiny run ({SMOKE['n_train']} train / {SMOKE['n_val']} val videos, "
                        f"{SMOKE['epochs']} epochs) to checkpoints/smoke/")
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args(argv)

    defaults = SMOKE if a.smoke else dict(n_train=TRAIN_SIZE, n_val=VAL_SIZE, epochs=MAX_EPOCHS)
    checkpoint = a.checkpoint or (f"checkpoints/smoke/gru_seed{a.seed}.pt" if a.smoke else "")
    cfg = TrainConfig(seed=a.seed, data=a.data,
                      n_train=a.n_train or defaults["n_train"], n_val=a.n_val or defaults["n_val"],
                      epochs=a.epochs or defaults["epochs"], batch_size=a.batch_size, lr=a.lr,
                      patience=a.patience, min_delta=a.min_delta, device=a.device,
                      checkpoint=checkpoint, cache_dir=a.cache_dir,
                      cache_format=a.cache_format, smoke=a.smoke,
                      log_every=a.log_every)
    train(cfg)


if __name__ == "__main__":
    main()
