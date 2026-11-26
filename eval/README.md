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
| `compute.json` | training and inference times — **fill this in by hand** |
| `test_eval.py` | 41 tests, including known-answer cases for every metric |
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

## Still to do

* `compute.json` needs real numbers, measured on one machine (see the
  `_machine` field) — everything else runs today.
* The team must decide when to open the test bank; until then use
  `--data starter` or the validation seed.

## Safety checks

A prediction file is refused if its video count or its `data_tag` does not
match the labels, because scoring predictions against the wrong videos
produces plausible-looking nonsense. Pass `--allow-tag-mismatch` only when you
know both refer to the same videos. One bad file is reported and skipped
rather than stopping the run.

Reference values to sanity-check against are in `baselines/HANDOFF.md`.
