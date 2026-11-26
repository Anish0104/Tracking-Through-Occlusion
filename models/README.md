# Recurrent SNN (`models/`)

Spiking neural network for tracking a moving disk through full occlusion.

## Files

- `lif.py` — `LIF` neuron (`beta=0.9`, `threshold=1.0`, reset by subtraction) and `FastSigmoidSpike` surrogate gradient (`slope=25`).
- `snn.py` — `RecurrentSNN` (1,070,274 trainable parameters).
- `overfit_test.py` — 32-video zero-occlusion sanity check.
- `train_snn.py` — full training loop matching `baselines/train_gru.py`.
- `predict_snn.py` — writes official test predictions (`snn_seed<k>.npz`, `snn_reset_seed<k>.npz`, and optional `snn_reset_every_seed<k>.npz`).
- `test_snn.py` — unit and contract tests (`python -m pytest -q models/test_snn.py`).

## Design choices

### Architecture (1,070,274 parameters)
- **Direct pixel current:** each `[B, 1, 64, 64]` frame is fed straight into `conv1` (no Poisson rate encoding), with one neural update per frame across all `T = 40` frames.
- **Spiking encoder:**
  - `Conv2d(1 -> 16, 3x3, pad=1)` -> `LIF` -> `MaxPool2d(2)` (`64x64 -> 32x32`, `160` params)
  - `Conv2d(16 -> 32, 3x3, pad=1)` -> `LIF` -> `MaxPool2d(2)` (`32x32 -> 16x16`, `4,640` params)
- **Recurrent LIF layer:** flatten (`8,192`) -> `Linear(8192 -> 128)` (`1,048,704` params) + `Linear(128 -> 128)` from the layer's own previous spikes (`16,512` params) -> `LIF` (`128` neurons).
- **Membrane readout:** `Linear(128 -> 2)` (`258` params) on the recurrent layer's membrane potential `mem3` followed by `sigmoid` gives smooth `(x, y)` coordinates in `(0, 1)`. Reading out `mem3` rather than binary spikes avoids quantized outputs.
- **Fixed LIF dynamics:** `beta=0.9` and `threshold=1.0` are fixed (0 trainable parameters), with subtraction reset (`mem = mem - spk * threshold`). Backpropagation through time uses a fast-sigmoid surrogate derivative `1 / (1 + 25 * |mem - threshold|)^2`.
- **State isolation:** all membrane states and previous recurrent spikes start at zero for every video and are never shared across videos. The model only ever receives `frames`.

### Training parity with CNN-GRU
`train_snn.py` matches `baselines/train_gru.py` on every shared hyperparameter (`CONTRACT.md` section 4):
- Loss: `common/loss.py: balanced_mse`
- Optimizer: `Adam(lr=1e-3)`, `batch_size=16`, `max_epochs=30`
- Early stopping: validation `balanced_mse` with `patience=5`, saving the best checkpoint to `checkpoints/snn_seed<k>.pt`
- Splits: train = `5,000` videos (`seed=100`), val = `1,000` videos (`seed=200`), loaded/cached via `baselines/data_cache.py`
- Training seeds: `0, 1, 2`

### Memory ablations (`snn_reset` vs `snn_reset_every`)
Both ablations run the exact same trained weights without retraining:
- **`snn_reset` (main ablation, `preds/snn_reset_seed<k>.npz`):** runs normally up to occlusion, resets `(mem1, mem2, mem3, spk3)` to zero **once** right before the first fully hidden frame (`visible_frac == 0`), and then continues the normal rollout. Videos with no hidden frames are never reset (identical to `snn`).
  - *Why onset is the main ablation:* resetting states at every frame starves LIF membrane integration across timesteps, dropping network activity from ~2,680 spikes/frame down to ~0.9 spikes/frame and making it impossible to separate lost temporal memory from a silenced network. Resetting once at occlusion onset erases only the pre-occlusion trajectory memory while letting LIF neurons keep integrating and firing during and after occlusion.
  - `visible_frac` is inspected only by `predict_snn.py` to pick the onset frame and is never passed to the model.
- **`snn_reset_every` (optional ablation, `preds/snn_reset_every_seed<k>.npz`):** resets all membrane states and recurrent spikes before every frame (`--reset-mode every_frame` or `--reset-mode both`).

## How to run

Quick checks and smoke test:
```bash
python -m models.lif
python -m models.snn
python -m models.overfit_test --epochs 60
python -m pytest -q models/test_snn.py
python -m models.train_snn --smoke --device cpu
```

Full training on Apple Silicon (`mps`) with shared data cache:
```bash
python -m baselines.data_cache --data generator --cache-dir checkpoints/data_cache
python -m models.train_snn --seed 0 --data generator --device mps --cache-dir checkpoints/data_cache
python -m models.train_snn --seed 1 --data generator --device mps --cache-dir checkpoints/data_cache
python -m models.train_snn --seed 2 --data generator --device mps --cache-dir checkpoints/data_cache
```

Write and verify official test-bank predictions (`preds/snn_seed<k>.npz` and `preds/snn_reset_seed<k>.npz`):
```bash
python -m models.predict_snn --seed 0 --device mps --verify
python -m models.predict_snn --seed 1 --device mps --verify
python -m models.predict_snn --seed 2 --device mps --verify
```
