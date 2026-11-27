"""Measure inference time for the compute table and write eval/compute.json.

    PYTHONPATH=. python -m eval.measure_compute                      # whatever runs here
    PYTHONPATH=. python -m eval.measure_compute --models kalman last_pos gru \
        --gru-checkpoint checkpoints/final_gru_generator_v1/gru_seed0.pt --device mps
    PYTHONPATH=. python -m eval.measure_compute --snn-checkpoint checkpoints/snn_seed0.pt

Inference time must come from one machine to be comparable (eval/README.md
item 4), so this records the machine alongside the numbers and refuses to mix
a new reading into timings taken elsewhere unless ``--force`` is passed.

Training time cannot be measured here: it belongs to whoever ran the training.
Pass it in with ``--train-s MODEL=SECONDS`` (repeatable), or edit the JSON.

This is the one place eval/ imports model code, and only to time it; the
report and figures still read prediction files alone.
"""

import argparse
import json
import os
import platform
import time

import numpy as np

DEFAULT_PATH = "eval/compute.json"
WARMUP = 1          # one untimed pass: the first is slow (imports, lazy init, device warm-up)
REPEATS = 3         # take the best of a few passes, which is the least noisy estimate


def machine_description(device="cpu"):
    """A short, reproducible note on what the numbers were measured on."""
    bits = [platform.platform(terse=True), platform.processor() or platform.machine(),
            f"python {platform.python_version()}", f"numpy {np.__version__}"]
    try:
        import torch
        bits.append(f"torch {torch.__version__}")
    except Exception:
        pass
    bits.append(f"device {device}")
    return ", ".join(b for b in bits if b)


def time_predict(predict, frames, repeats=REPEATS, warmup=WARMUP):
    """Best-of-N wall-clock milliseconds per video."""
    n = frames.shape[0]
    for _ in range(warmup):
        predict(frames)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict(frames)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0 / n


def measure(models, n=200, data_seed=300, device="cpu", gru_checkpoint=None,
            snn_checkpoint=None, repeats=REPEATS):
    """Time each model that can actually run here. -> ({model: ms}, {model: why not})"""
    from data.generator import make_batch
    frames = make_batch(n, data_seed)["frames"]
    timings, skipped = {}, {}
    for model in models:
        try:
            if model in ("snn", "snn_reset", "snn_reset_every"):
                predict = _snn_predictor(model, snn_checkpoint, device)
            else:
                from baselines.predict import _predictor
                predict, _ = _predictor(model, seed=0, checkpoint=gru_checkpoint,
                                        device=device, data="generator")
            timings[model] = time_predict(predict, frames, repeats=repeats)
        except Exception as exc:
            skipped[model] = f"{exc.__class__.__name__}: {exc}"
    return timings, skipped


def _snn_predictor(model, checkpoint, device):
    """Load the SNN for timing, through the team's own checkpoint loader.

    Uses models.train_snn.load_checkpoint (what models/predict_snn.py uses), so
    the timing path matches how the SNN is actually run.
    """
    import torch
    from models.train_snn import load_checkpoint, resolve_device
    path = checkpoint or "checkpoints/snn_seed0.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: pass --snn-checkpoint")
    dev = resolve_device(device)
    net, _ = load_checkpoint(path, dev)
    net.eval()
    reset_every = model == "snn_reset_every"

    def predict(frames):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(frames), dtype=torch.float32, device=dev)
            out, _ = net(x, reset_each_frame=reset_every)
            return out.cpu().numpy()
    return predict


def update_file(path, timings, machine, train_s=None, force=False):
    """Merge new readings into compute.json, keeping what is already there."""
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    old_machine = data.get("_machine", "")
    if (timings and old_machine and not old_machine.startswith("TODO")
            and old_machine != machine and not force):
        raise SystemExit(
            f"compute.json already holds timings from:\n  {old_machine}\nand this is:\n  {machine}\n"
            f"Inference times are only comparable from one machine. Re-measure every model "
            f"here, or pass --force if you know the mix is acceptable.")
    if timings:
        data["_machine"] = machine
    for model, ms in timings.items():
        entry = data.setdefault(model, {"train_s": None, "infer_ms_per_video": None})
        entry["infer_ms_per_video"] = round(ms, 4)
    for model, secs in (train_s or {}).items():
        entry = data.setdefault(model, {"train_s": None, "infer_ms_per_video": None})
        entry["train_s"] = secs
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data


def _parse_train(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--train-s wants MODEL=SECONDS, got {item!r}")
        model, value = item.split("=", 1)
        out[model.strip()] = float(value)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+",
                   default=["last_pos", "kalman", "gru", "snn", "snn_reset"])
    p.add_argument("--n", type=int, default=200, help="videos to time on")
    p.add_argument("--data-seed", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gru-checkpoint", default=None)
    p.add_argument("--snn-checkpoint", default=None)
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--train-s", nargs="+", default=None, metavar="MODEL=SECONDS",
                   help="record training seconds measured elsewhere")
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--force", action="store_true", help="allow mixing machines")
    a = p.parse_args(argv)

    timings, skipped = measure(a.models, a.n, a.data_seed, a.device,
                               a.gru_checkpoint, a.snn_checkpoint, a.repeats)
    machine = machine_description(a.device)
    data = update_file(a.out, timings, machine, _parse_train(a.train_s), a.force)

    print(f"timed on {a.n} videos: {machine}")
    for model, ms in sorted(timings.items()):
        print(f"  {model:16s} {ms:9.3f} ms/video")
    for model, why in sorted(skipped.items()):
        print(f"  {model:16s} not measured ({why})")
    print(f"wrote {a.out}")
    missing = [m for m, v in data.items()
               if not m.startswith("_") and v.get("train_s") is None]
    if missing:
        print(f"still missing training time for: {', '.join(sorted(missing))} "
              f"(use --train-s MODEL=SECONDS)")


if __name__ == "__main__":
    main()
