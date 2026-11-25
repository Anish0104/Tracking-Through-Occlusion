# Team contract

These rules let all four tasks run in parallel. Change them only with the whole team's agreement.

## 1. Data format

Every generator (`data/starter.py` now, `data/generator.py` later) returns a dict of numpy arrays:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `frames` | `[N, T, 64, 64]` | float32 in [0, 1] | grayscale video; disk = 1.0, barrier = 0.4, background = 0 |
| `xy` | `[N, T, 2]` | float32 | true disk center `(x, y)` divided by 64; pixel `i` has its center at `i + 0.5` |
| `visible_frac` | `[N, T]` | float32 in [0, 1] | fraction of disk pixels not covered by the barrier |

- A frame is **fully hidden** when `visible_frac == 0` and **fully visible** when `visible_frac == 1`. Anything in between is partially visible.
- Function signature: `make_batch(n, seed) -> dict`. The same seed must always give the same videos.
- `T = 40` by default.

## 2. What models may see

At inference, neural models get **only** `frames`, one frame at a time, in order. No future frames, no `xy`, no `visible_frac`, no velocity. The Kalman detector also uses pixels only.

## 3. Prediction files

Every model writes `preds/<model>_seed<k>.npz` using `common/io.py`:

- `pred_xy [N, T, 2]` in the same normalized units as `xy`
- `model`: one of `snn`, `snn_reset`, `gru`, `kalman`, `last_pos`
- `seed`: training seed (0 for non-learned baselines)
- `data_tag`: which test bank was used
- optional `spikes [N, T]` (SNN) and `n_params`

Evaluation reads only these files. It never imports model code.

## 4. Fixed values

| Item | Value |
|---|---|
| Training loss | `common/loss.py: balanced_mse` (both neural models) |
| Training seeds | 0, 1, 2 |
| Data seeds | train = 100, validation = 200, test = 300 |
| Split sizes | 5,000 / 1,000 / 1,000 videos |
| Parameter budget | 1,070,274 (the SNN layout in the proposal); the GRU must be within 10% and report its exact count |
| Success threshold | center error at most 5 pixels |
| Recovery | error at most 5 px for 3 frames in a row after full reappearance |
| Hidden durations | 0, 2, 4, 8 (train) and 12 (extrapolation test only) |

## 5. Git rules

- Work on your own branch (`data/...`, `snn/...`, `baselines/...`, `eval/...`), merge to `main` through a pull request.
- Only edit files in your own folder. Changes to `common/` or this file need the team's agreement.
- `python -m pytest -q` must pass before merging.
- Don't commit datasets, checkpoints or prediction files (see `.gitignore`).
