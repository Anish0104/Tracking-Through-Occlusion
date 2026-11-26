# Baselines → evaluation handoff

How the classical and neural baselines produce their test prediction files, and what
evaluation (eval/, owner Kunal) should compute from them.
Commit the files from the training machine that holds `checkpoints/final_gru_generator_v1/`.

## Official test bank

- Source: `data.generator`, split `test` = `SPLITS["test"]` = 1,000 videos, data seed **300**,
  hidden durations 0, 2, 4, 8, **12** (12 is extrapolation: never seen in training or validation).
- `make_batch(1000, 300)` and `iter_split("test")` give the same videos in the same order.
- **data_tag for every model: `generator_test_s300_v1`** (`data.generator.OFFICIAL_TEST_TAG`).
  `baselines/predict.py` sets it automatically for `--data generator --n 1000 --data-seed 300`
  and refuses that tag for any other n or seed. `iter_test_predictions` only accepts files with this tag
  and shape `[1000, 40, 2]`.

## Commands (run once, from the repository root, after the team agrees to open the test bank)

```bash
cd ~/code/Tracking-Through-Occlusion
export PYTHONPATH=$PWD
E=checkpoints/final_gru_generator_v1

.venv/bin/python -m baselines.predict --model last_pos --data generator
.venv/bin/python -m baselines.predict --model kalman   --data generator
for s in 0 1 2; do
  .venv/bin/python -m baselines.predict --model gru --data generator --seed $s \
    --checkpoint $E/gru_seed$s.pt --device mps
done
```

Defaults supply `--n 1000 --data-seed 300 --out-dir preds`. `--data generator` is required because the CLI defaults to the starter data.

Outputs (`common/io.save_preds`, CONTRACT.md §3):

| File | model | seed | n_params | Fitted on |
|---|---|---|---|---|
| `preds/last_pos_seed0.npz` | last_pos | 0 | – | nothing to fit |
| `preds/kalman_seed0.npz` | kalman | 0 | – | `baselines/kalman_config.json` (generator val seed 200) |
| `preds/gru_seed{0,1,2}.npz` | gru | 0/1/2 | 1,067,064 | `checkpoints/final_gru_generator_v1/gru_seed{0,1,2}.pt` |

Notes:

- **Data checks:** `predict.py` refuses to run if a checkpoint's `config.seed` differs from `--seed`. It also refuses if the GRU checkpoint or the Kalman config was fitted on a different generator: a different module, or a different source fingerprint (`8701d69abae7…`). Do not pass `--allow-data-mismatch` for the final files.
- **Overwriting:** `preds/last_pos_seed0.npz` and `preds/kalman_seed0.npz` currently exist from an early starter run (`data_tag` `starter_test_s300`), and the final run overwrites them. Archive them first if they are wanted.
- **Devices:** GRU predictions differ between CPU and MPS by at most 1.1e-5 px (measured on 200 validation videos), so either device is fine.
- **Inputs:** models receive frames only. Labels are never passed to them.

## What evaluation should compute (CONTRACT.md §4, eval/README.md)

Errors are center errors in pixels: the normalized difference × 64.

1. **Error by visibility category:** fully hidden (`visible_frac == 0`), fully visible (`== 1`) and partial. For each, take the mean per video, then the mean over videos that have such frames. Also report the all-frame mean.
2. **Success rate:** the fraction of frames with error ≤ 5 px.
3. **Recovery:** an error ≤ 5 px for 3 consecutive frames after full reappearance. Report the delay in frames and the fraction of videos that never recover.
4. **By hidden duration:** everything above for durations 0, 2, 4, 8 and 12, with 12 reported separately as extrapolation.
5. **Seeds:** per-seed results for the GRU (and SNN), with mean and spread over seeds 0, 1 and 2.
6. **Comparisons:** paired per-video comparisons between models, with bootstrap confidence intervals.
7. **Compute table:** parameters (GRU 1,067,064; SNN budget 1,070,274), training time and inference time.

Findings from validation that matter for how results are read (details in `checkpoints/final_gru_generator_v1/validation/README.md`):

- The 5 px success and recovery metrics are saturated on validation for GRU and Kalman: 100% success, every video recovers with delay 0. Model differences are sub-pixel, so the continuous pixel errors and paired CIs carry the comparison. A stricter recovery threshold (1–2 px) is worth reporting as a sensitivity check.
- GRU vs Kalman is split by category. The GRU is better on hidden and partial frames, Kalman on visible frames and overall. Report the categories, not only the overall mean.
- Kalman's hidden-frame error is about 2.5× larger on smooth-speed videos than on constant-speed ones; the GRU's is nearly unaffected. Motion type can be inferred from `xy`: a constant per-frame x step means constant speed.
- GRU seed spread (sd 0.03 px overall) is larger than one seed's sampling CI (about ±0.007 px), so always aggregate over seeds.

## Validation reference values (seed 200, for sanity-checking the eval code)

| Model | Overall | Visible | Partial | Hidden | ≤ 5 px |
|---|---|---|---|---|---|
| CNN-GRU (mean ± sd, 3 seeds) | 0.326 ± 0.031 | 0.284 ± 0.028 | 0.386 ± 0.045 | 0.509 ± 0.033 | 100% |
| Kalman (installed config) | 0.232 | 0.103 | 0.484 | 0.681 | 99.99% |
| Last position | 0.870 | 0.116 | 1.332 | 5.424 | 95.0% |

Reproduce with `PYTHONPATH=. python -m baselines.validate_final` (validation only) or
`python -m baselines.dev_metrics --data generator --models last_pos kalman gru --gru-checkpoint <ckpt> --gru-seed <k>`.
