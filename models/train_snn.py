"""Train the recurrent SNN (one model-initialization seed per run).

Matches baselines/train_gru.py on every shared setting (CONTRACT.md section 4):
    loss          common/loss.py: balanced_mse
    optimizer     Adam, lr 1e-3, batch 16, max 30 epochs
    early stop    validation balanced MSE, patience 5, keep best checkpoint
    data          train = 5,000 videos (seed 100), val = 1,000 videos (seed 200)
    model seeds   0, 1, 2

Examples:
    python -m models.train_snn --smoke --device cpu
    python -m baselines.data_cache --data generator --cache-dir checkpoints/data_cache
    python -m models.train_snn --seed 0 --device mps --cache-dir checkpoints/data_cache
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
from common.loss import balanced_mse
from models.snn import EXPECTED_PARAMS, build_model, count_params

TRAIN_SEED, VAL_SEED = 100, 200
TRAIN_SIZE, VAL_SIZE = 5000, 1000
LR, BATCH_SIZE, MAX_EPOCHS = 1e-3, 16, 30
PATIENCE = 5

SMOKE = dict(n_train=48, n_val=16, epochs=2)
DATA_KEYS = data_cache.DATA_KEYS


def load_split(source, n, seed, cache_dir=None, cache_format="auto"):
    """Load {frames, xy, visible_frac} from cache_dir (memmap) or directly in RAM."""
    if cache_dir is None:
        batch = data_cache.resolve_module(source).make_batch(n=n, seed=seed)
        return {k: batch[k] for k in DATA_KEYS}
    return data_cache.get_split(source, n, seed, cache_dir, fmt=cache_format)


def check_split(split):
    n, t = split["frames"].shape[:2]
    assert split["frames"].shape[2:] == (64, 64), split["frames"].shape
    assert split["xy"].shape == (n, t, 2), split["xy"].shape
    assert split["visible_frac"].shape == (n, t), split["visible_frac"].shape


def _take(arr, idx):
    """Index arr in idx order; reads disk-backed caches sequentially for speed."""
    if type(arr) is np.ndarray:
        return arr[idx]
    perm = np.argsort(idx, kind="stable")
    out = np.asarray(arr[idx[perm]])
    return out[np.argsort(perm, kind="stable")]


def iterate_batches(split, batch_size, order=None):
    """Yield (frames, xy, visible_frac) torch tensors one batch at a time."""
    n = len(split["frames"])
    order = np.arange(n) if order is None else np.asarray(order)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        yield tuple(
            torch.from_numpy(np.ascontiguousarray(_take(split[k], idx), np.float32))
            for k in DATA_KEYS
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name):
    """'cpu', 'cuda', 'mps' or 'auto' (cuda > mps > cpu)."""
    if name == "auto":
        name = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    return torch.device(name)


class EarlyStopping:
    """Stop after `patience` epochs without validation loss improving."""

    def __init__(self, patience=PATIENCE, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.best_epoch = None
        self.bad_epochs = 0

    def update(self, loss, epoch):
        if loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = loss, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self):
        return self.patience is not None and self.bad_epochs >= self.patience


@dataclass
class TrainConfig:
    seed: int = 0
    data: str = "generator"
    n_train: int = TRAIN_SIZE
    n_val: int = VAL_SIZE
    train_seed: int = TRAIN_SEED
    val_seed: int = VAL_SEED
    epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    patience: int = PATIENCE
    min_delta: float = 0.0
    device: str = "auto"
    checkpoint: str = ""
    cache_dir: str = ""
    cache_format: str = "auto"
    smoke: bool = False
    log_every: int = 50

    def __post_init__(self):
        if self.train_seed == self.val_seed:
            raise ValueError("training and validation data seeds must differ")
        if 300 in (self.train_seed, self.val_seed):
            raise ValueError("data seed 300 is the held-out test bank; never train or validate on it")
        if not self.checkpoint:
            self.checkpoint = f"checkpoints/snn_seed{self.seed}.pt"


def train_one_epoch(model, split, optimizer, batch_size, device, rng, log_every=0):
    """One epoch of BPTT over all 40 frames in seeded random batch order."""
    model.train()
    order = rng.permutation(len(split["frames"]))
    total_loss, total_spikes, count = 0.0, 0.0, 0

    for step, (frames, xy, vis) in enumerate(iterate_batches(split, batch_size, order), 1):
        frames, xy, vis = frames.to(device), xy.to(device), vis.to(device)
        pred, spikes = model(frames)                # model sees frames only
        loss = balanced_mse(pred, xy, vis)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = len(frames)
        total_loss += loss.item() * bs
        total_spikes += spikes.mean().item() * bs
        count += bs
        if log_every and step % log_every == 0:
            print(
                f"    step {step:4d}  train loss {total_loss / count:.6f}  "
                f"spk/frame {total_spikes / count:6.1f}",
                flush=True,
            )
    return total_loss / count, total_spikes / count


@torch.no_grad()
def _probe_max_abs_mem(model, frames):
    """Run one batch step-by-step to measure peak |membrane potential| across layers."""
    state = model.init_state(frames.shape[0], frames.device)
    max_mem = 0.0
    for t in range(frames.shape[1]):
        _, state, _ = model.step(frames[:, t], state)
        mem1, mem2, mem3, _ = state
        step_max = max(
            mem1.abs().max().item(),
            mem2.abs().max().item(),
            mem3.abs().max().item(),
        )
        if step_max > max_mem:
            max_mem = step_max
    return max_mem


@torch.no_grad()
def evaluate(model, split, batch_size, device):
    """Mean balanced MSE, spikes/frame, and peak |mem| on a split."""
    was_training = model.training
    model.eval()
    total_loss, total_spikes, count = 0.0, 0.0, 0
    max_abs_mem = 0.0

    for batch_idx, (frames, xy, vis) in enumerate(iterate_batches(split, batch_size)):
        frames_dev = frames.to(device)
        pred, spikes = model(frames_dev)            # model sees frames only
        loss = balanced_mse(pred, xy.to(device), vis.to(device))
        bs = len(frames)
        total_loss += loss.item() * bs
        total_spikes += spikes.mean().item() * bs
        count += bs
        if batch_idx == 0:
            max_abs_mem = _probe_max_abs_mem(model, frames_dev)

    model.train(was_training)
    return total_loss / count, total_spikes / count, max_abs_mem


def save_checkpoint(path, model, optimizer, epoch, val_loss, history, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "snn",
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
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    """Load a saved RecurrentSNN checkpoint. Returns (model in eval mode, payload)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model") != "snn":
        raise ValueError(f"{path} is not an SNN checkpoint")
    model = build_model(seed=0)
    model.load_state_dict(payload["model_state"])
    n = count_params(model)
    if n != payload["n_params"] or n != EXPECTED_PARAMS:
        raise ValueError(f"parameter count mismatch: model {n}, checkpoint {payload['n_params']}")
    return model.to(device).eval(), payload


def train(cfg: TrainConfig, train_split=None, val_split=None):
    """Train one seed and write best checkpoint + JSON summary."""
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
    shuffle_rng = np.random.default_rng(cfg.seed)
    stopper = EarlyStopping(cfg.patience if cfg.patience > 0 else None, cfg.min_delta)
    data_module = data_cache.resolve_module(cfg.data)
    saved_config = {
        **asdict(cfg),
        "data_module": data_module.__name__,
        "data_fingerprint": data_cache.source_fingerprint(data_module),
    }

    print(
        f"RecurrentSNN seed {cfg.seed} | {n_params:,} params | device {device} | "
        f"data {data_module.__name__} | "
        f"train {len(train_split['frames'])} (data seed {cfg.train_seed}) | "
        f"val {len(val_split['frames'])} (data seed {cfg.val_seed})",
        flush=True,
    )

    history = []
    stopped_early = False
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_spk = train_one_epoch(
            model, train_split, optimizer, cfg.batch_size, device, shuffle_rng, cfg.log_every
        )
        val_loss, val_spk, max_mem = evaluate(model, val_split, cfg.batch_size, device)
        improved = stopper.update(val_loss, epoch)
        elapsed = round(time.time() - t0, 2)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_spikes_per_frame": train_spk,
            "val_spikes_per_frame": val_spk,
            "val_max_abs_mem": max_mem,
            "seconds": elapsed,
        })
        if improved:
            save_checkpoint(cfg.checkpoint, model, optimizer, epoch, val_loss, history, saved_config)

        warn = ""
        if val_spk == 0.0:
            warn += "  [WARN: zero spikes]"
        if max_mem > 100.0 or not math.isfinite(max_mem):
            warn += f"  [WARN: high membrane {max_mem:.1f}]"

        print(
            f"epoch {epoch:2d}  train {train_loss:.6f}  val {val_loss:.6f}  "
            f"spk/f {val_spk:6.1f}  max|mem| {max_mem:5.2f}"
            f"{'  *best, saved' if improved else ''}  ({elapsed}s){warn}",
            flush=True,
        )
        if stopper.should_stop:
            stopped_early = True
            print(f"early stop: no val improvement for {stopper.patience} epochs", flush=True)
            break

    summary = {
        "seed": cfg.seed,
        "checkpoint": cfg.checkpoint,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "n_params": n_params,
        "data_fingerprint": saved_config["data_fingerprint"],
        "history": history,
        "config": asdict(cfg),
    }
    Path(cfg.checkpoint).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"best val {stopper.best:.6f} at epoch {stopper.best_epoch} -> {cfg.checkpoint}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="model initialization seed (0, 1 or 2)")
    p.add_argument(
        "--data",
        default="generator",
        help="data source: generator, starter, or a module path with make_batch(n, seed)",
    )
    p.add_argument("--n-train", type=int, default=None, help=f"default {TRAIN_SIZE}")
    p.add_argument("--n-val", type=int, default=None, help=f"default {VAL_SIZE}")
    p.add_argument("--epochs", type=int, default=None, help=f"maximum epochs, default {MAX_EPOCHS}")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--patience", type=int, default=PATIENCE, help="early-stopping patience; 0 disables")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--checkpoint", default="", help="default: checkpoints/snn_seed<seed>.pt")
    p.add_argument(
        "--cache-dir",
        default="",
        help="shared on-disk cache for the splits (baselines/data_cache.py); default: in memory",
    )
    p.add_argument("--cache-format", default="auto", choices=data_cache.FORMATS)
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"tiny run ({SMOKE['n_train']} train / {SMOKE['n_val']} val videos, "
             f"{SMOKE['epochs']} epochs) to checkpoints/smoke/",
    )
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args(argv)

    defaults = SMOKE if a.smoke else dict(n_train=TRAIN_SIZE, n_val=VAL_SIZE, epochs=MAX_EPOCHS)
    checkpoint = a.checkpoint or (f"checkpoints/smoke/snn_seed{a.seed}.pt" if a.smoke else "")
    cfg = TrainConfig(
        seed=a.seed,
        data=a.data,
        n_train=a.n_train or defaults["n_train"],
        n_val=a.n_val or defaults["n_val"],
        epochs=a.epochs or defaults["epochs"],
        batch_size=a.batch_size,
        lr=a.lr,
        patience=a.patience,
        min_delta=a.min_delta,
        device=a.device,
        checkpoint=checkpoint,
        cache_dir=a.cache_dir,
        cache_format=a.cache_format,
        smoke=a.smoke,
        log_every=a.log_every,
    )
    train(cfg)


if __name__ == "__main__":
    main()
