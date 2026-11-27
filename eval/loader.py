"""Finding prediction files and pairing them with the matching labels.

Evaluation reads ``preds/*.npz`` only (CONTRACT.md section 3) and never imports
model code, so eval/ does not need torch or snntorch installed.

Two label sources are supported:

* ``generator`` - the official test bank (``data.generator``, 1,000 videos,
  seed 300, durations 0/2/4/8/12). This is what the final report uses.
* ``starter`` / ``fake`` - the small starter generator, for building and
  testing the pipeline before real predictions exist (eval/README.md).

A prediction file carries its own ``data_tag``, so we check it against the
label source and refuse to score mismatched pairs rather than silently
producing meaningless numbers.
"""

import glob
import os
import re

import numpy as np

from common.io import load_preds

#: Spiking neurons whose spikes are summed into the ``spikes`` array
#: (models/snn.py: lif1 16x64x64, lif2 32x32x32 counted before pooling,
#: lif3 128). Used for spikes per neuron per frame.
SNN_NEURONS = 16 * 64 * 64 + 32 * 32 * 32 + 128      # 98,432

#: Order models appear in tables and figures; unknown models are appended.
MODEL_ORDER = ("snn", "snn_reset", "snn_reset_every", "gru", "kalman", "last_pos")

_FILENAME = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)\.npz$")


def model_sort_key(model):
    """Sort key putting known models in MODEL_ORDER, then unknown ones by name."""
    try:
        return (0, MODEL_ORDER.index(model), model)
    except ValueError:
        return (1, 0, model)


def find_pred_files(preds_dir="preds", pattern="*.npz"):
    """Prediction files in a directory, sorted by model then seed."""
    files = sorted(glob.glob(os.path.join(preds_dir, pattern)))
    def key(p):
        m = _FILENAME.match(os.path.basename(p))
        name = m.group("model") if m else os.path.basename(p)
        seed = int(m.group("seed")) if m else -1
        return model_sort_key(name) + (seed,)
    return sorted(files, key=key)


def load_labels(source="generator", n=None, seed=None):
    """Labels for a data source. -> dict with xy, visible_frac, tag, (frames)

    ``frames`` is included because demo/animate.py needs the video itself.
    """
    if source in ("generator", "test"):
        from data.generator import OFFICIAL_TEST_TAG, SPLITS, make_batch
        size, data_seed, _ = SPLITS["test"]
        n = size if n is None else n
        seed = data_seed if seed is None else seed
        b = make_batch(n, seed)
        tag = OFFICIAL_TEST_TAG if (n, seed) == (size, data_seed) else f"generator_s{seed}_n{n}"
    elif source in ("starter", "fake"):
        from data.starter import make_batch
        n = 64 if n is None else n
        seed = 300 if seed is None else seed
        b = make_batch(n=n, seed=seed)
        tag = f"starter_test_s{seed}"
    else:
        raise ValueError(f"unknown label source {source!r}; use generator or starter")
    return {"frames": b["frames"], "xy": b["xy"], "visible_frac": b["visible_frac"],
            "tag": tag, "n": int(b["xy"].shape[0]), "seed": seed, "source": source}


def load_prediction(path, labels, strict_tag=True):
    """Load one prediction file and check it lines up with ``labels``.

    Raises ValueError on a shape or tag mismatch: scoring predictions against
    labels from different videos would produce numbers that look plausible and
    mean nothing. ``strict_tag=False`` downgrades the tag check to a warning
    string, which the development workflow on starter data uses.
    """
    p = load_preds(path)
    pred = np.asarray(p["pred_xy"], np.float64)
    want = (labels["n"], labels["xy"].shape[1], 2)
    if pred.shape != want:
        raise ValueError(f"{os.path.basename(path)}: pred_xy is {pred.shape}, "
                         f"but the labels are {want}. Scoring a different bank "
                         f"than the predictions were made on is never valid.")
    warning = None
    if p["data_tag"] != labels["tag"]:
        msg = (f"{os.path.basename(path)}: data_tag {p['data_tag']!r} does not match "
               f"the label source {labels['tag']!r}")
        if strict_tag:
            raise ValueError(msg + ". Pass --allow-tag-mismatch only when you know "
                                   "both refer to the same videos.")
        warning = msg
    return {
        "path": path,
        "model": p["model"],
        "seed": p["seed"],
        "data_tag": p["data_tag"],
        "pred_xy": pred,
        "spikes": np.asarray(p["spikes"], np.float64) if "spikes" in p else None,
        "n_params": int(p["n_params"]) if "n_params" in p else None,
        "warning": warning,
    }


def load_all(preds_dir="preds", source="generator", strict_tag=True, n=None, seed=None):
    """Load every prediction file alongside its labels.

    Returns ``(labels, entries, problems)``; files that cannot be scored go to
    ``problems`` as ``(path, message)`` so one bad file does not sink the run.
    """
    labels = load_labels(source, n=n, seed=seed)
    entries, problems = [], []
    for path in find_pred_files(preds_dir):
        try:
            entries.append(load_prediction(path, labels, strict_tag=strict_tag))
        except Exception as exc:                      # keep going; report at the end
            problems.append((path, str(exc)))
    entries.sort(key=lambda e: model_sort_key(e["model"]) + (e["seed"],))
    return labels, entries, problems


def make_fake_predictions(labels, models=("snn", "gru", "kalman", "last_pos"), seeds=(0, 1, 2),
                          out_dir="preds"):
    """Write placeholder prediction files so the pipeline can be run end to end.

    eval/README.md: build on ``common/fake.py`` now, real files plug in later
    with no changes. Each model gets a different hidden-frame noise level so
    the tables, figures and comparisons have something to separate; the SNN
    also gets a ``spikes`` array. These are **not** results.
    """
    from common.fake import fake_preds
    from common.io import save_preds

    # hidden_px: how far predictions drift while the target is hidden.
    profiles = {
        "snn":          dict(visible_px=0.8, hidden_px=2.0, spikes=True, n_params=1_070_274),
        "snn_reset":    dict(visible_px=0.8, hidden_px=4.5, spikes=True, n_params=1_070_274),
        "gru":          dict(visible_px=0.9, hidden_px=2.6, spikes=False, n_params=1_067_064),
        "kalman":       dict(visible_px=0.5, hidden_px=3.4, spikes=False, n_params=None),
        "last_pos":     dict(visible_px=0.4, hidden_px=9.0, spikes=False, n_params=None),
    }
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    written = []
    for model in models:
        prof = profiles.get(model, dict(visible_px=1.0, hidden_px=4.0, spikes=False, n_params=None))
        # Non-learned baselines have a single run: seed 0 (CONTRACT.md section 3).
        model_seeds = (0,) if model in ("kalman", "last_pos") else seeds
        for seed in model_seeds:
            pred = fake_preds(labels["xy"], labels["visible_frac"], seed=1000 + seed,
                              visible_px=prof["visible_px"], hidden_px=prof["hidden_px"])
            spikes = None
            if prof["spikes"]:
                # Plausible activity: a little higher while the target is visible.
                base = 2600.0 + 40.0 * seed
                spikes = (base * (0.9 + 0.2 * labels["visible_frac"])
                          + rng.normal(0, 25, labels["visible_frac"].shape)).astype(np.float32)
            path = os.path.join(out_dir, f"{model}_seed{seed}.npz")
            save_preds(path, pred, model=model, seed=seed, data_tag=labels["tag"],
                       spikes=spikes, n_params=prof["n_params"])
            written.append(path)
    return written
