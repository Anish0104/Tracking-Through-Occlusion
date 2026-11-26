# Video generator

Use the full generator anywhere the starter's batch dictionary is expected.

```python
from data.generator import iter_split, make_batch

example = make_batch(8, seed=1, durations=(4,), motion="constant")
for batch in iter_split("train", batch_size=16):
    frames = batch["frames"]  # [N, 40, 64, 64]
    xy = batch["xy"]          # normalized disk centers [N, 40, 2]
    visible = batch["visible_frac"]  # visible disk fraction [N, 40]
```

All arrays are float32. For the official banks use iter_split. Changing batch size
preserves both the videos and their order. Use frames alone for tracker inference.

| Split | Videos | Seed | Fully hidden frames |
|---|---:|---:|---|
| train | 5000 | 100 | 0, 2, 4, 8 |
| val | 1000 | 200 | 0, 2, 4, 8 |
| test | 1000 | 300 | 0, 2, 4, 8, 12 |

We balance counts across duration and mean horizontal speed (0.72–0.87,
0.95–1.10, or 1.18–1.32 pixels/frame). Mixed motion independently samples
constant or sinusoidally varying horizontal velocity with equal probability.
Both modes include vertical drift. Paths stay in bounds and never bounce.

For a hidden interval we sample its start, then choose the barrier columns
that cover the actual disk pixels throughout the requested interval. We reject
additional fully hidden frames or fewer than four fully visible frames at
either end. Fully visible controls have an off-path barrier. Position labels
continue through occlusion. Visibility uses disk pixel counts before and
after the barrier is drawn.

Run from the repository root:

```bash
python -m pytest -q
python -m data.check_splits
```

The audit streams all three splits, checks balance and visibility, and checks
the complete normalized paths for duplicates. No generated videos need to be
committed or downloaded.

## Official test predictions and evaluation

Use `iter_split("test")` for both inference and evaluation. Keep predictions
in iteration order and save them with `data.generator.OFFICIAL_TEST_TAG`
(`generator_test_s300_v1`). The compatibility call `make_batch(1000, 300)`
returns exactly the same frames, centers, and visibility labels. Passing
explicit durations or a different motion mode selects a custom experiment.
To preview the first ten official videos, use
`next(iter_split("test", batch_size=10))`. A custom ten-video batch has its
own shuffle.

With the team's baseline prediction CLI, generate the official last-position
predictions from the repository root:

```bash
python -m baselines.predict --model last_pos --data generator --data-tag generator_test_s300_v1
```

Supply the same data source and tag for every official model. Regenerate
predictions previously written with the old defaults. Before scoring, check
that the saved tag equals OFFICIAL_TEST_TAG and that pred_xy has shape
(1000, 40, 2). Match its successive rows to the labels from iter_split("test").
Reserve this tag for predictions made on the complete, unmodified test bank.

Evaluation can perform these checks and consume aligned labels directly:

```python
from data.generator import iter_test_predictions

for pred_xy, true_xy, visible_frac in iter_test_predictions("preds/last_pos_seed0.npz"):
    error_px = ((pred_xy - true_xy) ** 2).sum(axis=-1) ** 0.5 * 64
```
