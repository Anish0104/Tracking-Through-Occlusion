# eval/ and demo/ (owner: Kunal)

Metrics, statistics, figures and the animation. Evaluation reads
`preds/*.npz` only (CONTRACT.md section 3) and never imports model code, so it
runs with just numpy and matplotlib — no torch, no snntorch, no checkpoints.

## Quick start

```bash
export PYTHONPATH=$PWD

# Real results, once the prediction files are in preds/ (see baselines/HANDOFF.md)
python -m eval.report  --data generator --json results.json --out report.txt
python -m eval.figures --data generator
python -m demo.animate --data generator --pick worst

# No prediction files yet? Build and check the pipeline on placeholder data.
python -m eval.report --data starter --n 200 --make-fake
```

`--make-fake` writes placeholder predictions from `common/fake.py`. They are a
scaffold for checking that the pipeline runs, **not results**.

## Files

| File | What it does |
|---|---|
| `metrics.py` | error, categories, success, recovery, spike activity |
| `stats.py` | bootstrap confidence intervals, paired comparisons, seed spread |
| `loader.py` | finds prediction files, pairs them with labels, checks they match |
| `report.py` | every table; `--json` for machine-readable results |
| `figures.py` | the five figures below, into `figures/` |
| `measure_compute.py` | measures inference time, writes `compute.json` |
| `compute.json` | parameters/timings for the compute table |
| `test_eval.py` | 48 tests, including known-answer cases for every metric |
| `../demo/animate.py` | side-by-side animation, including failure cases |

## Definitions

* **Error** is the center error in pixels: the normalized difference x 64.
* **Categories** come from `visible_frac`: fully visible (`== 1`), fully hidden
  (`== 0`), partial (in between).
* **Aggregation** is mean per video, then over videos that have such a frame.
  A video without a frame of some category is dropped from that column rather
  than counted as zero. This matches `baselines/dev_metrics.py`.
* **Success** is error <= 5 px; **recovery** is error <= 5 px for 3 frames in a
  row after the target is *fully* visible again (CONTRACT.md section 4).
* **Hidden duration** is not stored in the batch dict, so it is recovered from
  the labels as the number of fully hidden frames. The generator uses one
  contiguous hidden interval, so this equals the requested duration.
* **Seeds**: a model's seeds are averaged per video before models are compared,
  so a 3-seed model does not outweigh a 1-seed baseline.

### Two choices worth knowing about

**Recovery excludes videos that cannot show it.** If a video ends while the
target is still hidden, or ends fewer than 3 frames after it reappears, no
tracker can produce the required run of 3 good frames. Counting those as
"never recovered" would put the same floor under every model: on the starter
bank a *perfect* tracker scored 13.6% "never recovered" that way. They are
excluded from the denominator instead, and `test_perfect_tracker_never_fails_to_recover`
guards this.

**The 5 px metrics saturate.** As `baselines/HANDOFF.md` predicted, every model
except last-position recovers instantly at 5 px, so the report also prints a
strict 1 px recovery column, and the comparison rests on the continuous errors
and the paired intervals rather than on the success rate.

## Report sections

1. error by category and success at 5 px, per seed, with mean +/- sd over seeds
2. everything by hidden duration (0, 2, 4, 8, 12; **12 is marked `*`** because
   it is extrapolation and was never trained on)
3. recovery delay and never-recovered, at 5 px and at a strict 1 px
4. bootstrap confidence intervals for each model's error
5. paired per-video comparisons with bootstrap CIs, including the SNN vs
   `snn_reset` state-reset ablation
6. error split by motion type (constant vs smooth), inferred from the labels:
   a constant per-frame step in x means constant speed
7. spiking activity: spikes per frame and per neuron per frame (98,432 spiking
   neurons: `lif1` 16x64x64 + `lif2` 32x32x32 + `lif3` 128)
8. compute: parameters from the prediction files, times from `compute.json`

## Figures (`figures/`, gitignored)

| File | Shows |
|---|---|
| `error_vs_duration.png` | error vs hidden duration, hidden and all frames |
| `recovery_vs_duration.png` | recovery delay and never-recovered vs duration |
| `trajectories.png` | predicted vs true paths, with the hidden span shaded |
| `error_over_time.png` | error aligned on the moment of full reappearance |
| `error_by_category.png` | per-category error with bootstrap intervals |

## Animation

```bash
python -m demo.animate --pick worst    # failure cases (the default mix shows both)
python -m demo.animate --videos 3 17 42 --format mp4
```

Each panel shows the frame as the models see it, the true path, one marker per
model, and the live per-model pixel error in the title. Falls back to GIF if
ffmpeg is missing.

## Compute table

```bash
# Inference time, on whichever models can run on this machine
python -m eval.measure_compute --models last_pos kalman gru snn \
    --gru-checkpoint checkpoints/final_gru_generator_v1/gru_seed0.pt \
    --snn-checkpoint checkpoints/snn_seed0.pt --device mps

# Training time has to come from whoever trained it
python -m eval.measure_compute --train-s gru=4200 snn=9100
```

Inference times are only comparable when measured on one machine, so the tool
records the machine in `_machine` and refuses to mix a second one into the file
unless you pass `--force`. Models whose checkpoints are not present are skipped
with a reason rather than failing the run.

`last_pos` and `kalman` are recorded with `train_s: 0`: neither has a training
phase (Kalman is *tuned* on validation). The SNN and GRU still need their
training seconds and inference times from the machines that trained them.

## Still to do

* SNN/GRU training seconds and inference times (needs the checkpoints and the
  training machines) — everything else runs today.
* Final numbers once the three SNN seeds and the three GRU seeds are trained
  and the official predictions are generated.

## Safety checks

A prediction file is refused if its video count or its `data_tag` does not
match the labels, because scoring predictions against the wrong videos
produces plausible-looking nonsense. Pass `--allow-tag-mismatch` only when you
know both refer to the same videos. One bad file is reported and skipped
rather than stopping the run.

Reference values to sanity-check against are in `baselines/HANDOFF.md`.

## Cross-check against baselines/dev_metrics.py

The metrics here were checked against Anuj's independently written
`baselines/dev_metrics.py` on the validation bank (seed 200): overall, visible,
partial and hidden error agree to 10 decimal places for both `last_pos` and
`kalman`, and reproduce the reference table in `baselines/HANDOFF.md`. The
motion split also independently reproduces the handoff's finding that Kalman's
hidden-frame error is about 2.5x worse on smooth motion (measured here: 2.55x)
while last-position is unaffected.
