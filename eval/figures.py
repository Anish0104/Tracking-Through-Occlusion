"""Figures for the report (eval/README.md).

    PYTHONPATH=. python -m eval.figures                       # official test bank
    PYTHONPATH=. python -m eval.figures --data starter --n 200 --preds-dir preds

Produces, in figures/:
    error_vs_duration.png      hidden-frame and all-frame error vs hidden duration
    recovery_vs_duration.png   recovery delay and never-recovered vs hidden duration
    trajectories.png           predicted vs true paths, with the occluded span shaded
    error_over_time.png        error against frame index, aligned on reappearance
    error_by_category.png      per-category error per model

Uses the non-interactive Agg backend so it runs over ssh and in CI. PNGs are
gitignored; regenerate them with this script.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval import stats
from eval.loader import load_all, make_fake_predictions
from eval.metrics import (CATEGORIES, SUCCESS_PX, category_masks, hidden_durations,
                          per_video_mean, pixel_error, reappearance_index, recovery_delay)
from eval.report import EXTRAPOLATION_DURATION, build_runs, duration_breakdown, group_by_model

#: One stable colour and marker per model, so a model looks the same in every figure.
STYLE = {
    "snn":              ("#1f77b4", "o", "-"),
    "snn_reset":        ("#17becf", "v", "--"),
    "snn_reset_every":  ("#9edae5", "^", ":"),
    "gru":              ("#ff7f0e", "s", "-"),
    "kalman":           ("#2ca02c", "D", "-."),
    "last_pos":         ("#d62728", "x", ":"),
}
FALLBACK = ("#7f7f7f", ".", "-")
DPI = 150


def style(model):
    return STYLE.get(model, FALLBACK)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _mark_extrapolation(ax, durations):
    """Shade the extrapolation duration so it is never read as a trained result."""
    if EXTRAPOLATION_DURATION in durations and len(durations) > 1:
        step = min(np.diff(sorted(durations))) if len(durations) > 1 else 1
        ax.axvspan(EXTRAPOLATION_DURATION - step / 2, EXTRAPOLATION_DURATION + step / 2,
                   color="0.9", zorder=0)
        ax.text(EXTRAPOLATION_DURATION, ax.get_ylim()[1], " extrapolation",
                fontsize=7, va="top", ha="center", color="0.35", rotation=90)


def fig_error_vs_duration(by_model, data, out_dir):
    """Error against hidden duration: the headline figure."""
    breakdown = duration_breakdown(by_model, data)
    durs = sorted(next(iter(breakdown.values())).keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric, title in ((axes[0], "hidden_px", "hidden frames"),
                              (axes[1], "err_px", "all frames")):
        for model, per in breakdown.items():
            c, m, ls = style(model)
            y = np.array([per[d][metric][0] for d in durs], float)
            e = np.array([per[d][metric][1] for d in durs], float)
            e = np.where(np.isfinite(e), e, 0.0)
            ok = np.isfinite(y)
            if not ok.any():
                continue
            ax.errorbar(np.array(durs)[ok], y[ok], yerr=e[ok], label=model, color=c,
                        marker=m, ls=ls, capsize=2, ms=5, lw=1.6)
        ax.axhline(SUCCESS_PX, color="0.6", lw=0.8, ls=":")
        ax.text(durs[0], SUCCESS_PX, f" {SUCCESS_PX:g} px success threshold",
                fontsize=7, va="bottom", color="0.4")
        ax.set_title(f"center error, {title}")
        ax.set_xlabel("hidden duration (frames)")
        ax.set_xticks(durs)
        ax.grid(alpha=0.3)
        _mark_extrapolation(ax, durs)
    axes[0].set_ylabel("center error (px)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Error vs hidden duration (error bars: sd over training seeds)", fontsize=11)
    return _save(fig, out_dir, "error_vs_duration.png")


def fig_recovery_vs_duration(by_model, data, out_dir):
    """Recovery delay and never-recovered fraction against hidden duration."""
    breakdown = duration_breakdown(by_model, data)
    durs = [d for d in sorted(next(iter(breakdown.values())).keys()) if d > 0]
    if not durs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for model, per in breakdown.items():
        c, m, ls = style(model)
        for ax, key, scale in ((axes[0], "delay", 1.0), (axes[1], "never", 100.0)):
            y = np.array([per[d][key][0] for d in durs], float) * scale
            ok = np.isfinite(y)
            if ok.any():
                ax.plot(np.array(durs)[ok], y[ok], label=model, color=c, marker=m, ls=ls,
                        ms=5, lw=1.6)
    axes[0].set_ylabel("recovery delay (frames)")
    axes[0].set_title("frames until error stays <= 5 px for 3 frames")
    axes[1].set_ylabel("never recovered (%)")
    axes[1].set_title("videos that never recover")
    for ax in axes:
        ax.set_xlabel("hidden duration (frames)")
        ax.set_xticks(durs)
        ax.grid(alpha=0.3)
        _mark_extrapolation(ax, durs)
    axes[0].legend(fontsize=8)
    fig.suptitle("Recovery after the target reappears", fontsize=11)
    return _save(fig, out_dir, "recovery_vs_duration.png")


def fig_trajectories(labels, by_model, out_dir, n_show=3, worst_model=None):
    """Predicted vs true paths: the clearest picture of what each model does.

    Shows the videos with the longest occlusion, and marks the hidden span, so
    the drift during occlusion and the snap back afterwards are both visible.
    """
    vf = labels["visible_frac"]
    dur = hidden_durations(vf)
    order = np.argsort(-dur)
    picks = order[:n_show]
    fig, axes = plt.subplots(2, len(picks), figsize=(4.0 * len(picks), 7.0),
                             squeeze=False)
    size = 64
    for col, vid in enumerate(picks):
        hidden = np.flatnonzero(vf[vid] <= 0)
        # top row: path in the image plane
        ax = axes[0][col]
        ax.imshow(labels["frames"][vid][hidden[0] if hidden.size else 0],
                  cmap="gray", origin="upper", extent=(0, size, size, 0), vmin=0, vmax=1)
        true = labels["xy"][vid] * size
        ax.plot(true[:, 0], true[:, 1], color="w", lw=2.4, label="true", zorder=3)
        ax.plot(true[:, 0], true[:, 1], color="k", lw=1.0, ls="--", zorder=4)
        # Models often overlap almost exactly when visible; decreasing width and
        # alpha down the stack keeps every curve findable instead of hiding the
        # ones drawn first.
        n_models = len(by_model)
        for k, (model, runs) in enumerate(by_model.items()):
            c, m, ls = style(model)
            p = np.mean([r["pred"][vid] for r in runs], axis=0) * size
            ax.plot(p[:, 0], p[:, 1], color=c, lw=2.2 - 1.1 * k / max(n_models - 1, 1),
                    ls=ls, alpha=0.9, zorder=5 + k, label=model)
            if hidden.size:
                ax.plot(p[hidden, 0], p[hidden, 1], color=c, marker=m, ms=3.5, ls="none",
                        zorder=10 + k)
        ax.set_title(f"video {vid}: {dur[vid]} hidden"
                     + ("  (extrap.)" if dur[vid] == EXTRAPOLATION_DURATION else ""),
                     fontsize=9)
        ax.set_xlim(0, size); ax.set_ylim(size, 0)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)

        # bottom row: x against time, where the occlusion gap is easiest to read
        ax = axes[1][col]
        t = np.arange(true.shape[0])
        if hidden.size:
            ax.axvspan(hidden[0] - 0.5, hidden[-1] + 0.5, color="0.88",
                       label="fully hidden", zorder=0)
        ax.plot(t, true[:, 0], color="k", lw=2.4, label="true x", zorder=3)
        for k, (model, runs) in enumerate(by_model.items()):
            c, m, ls = style(model)
            p = np.mean([r["pred"][vid] for r in runs], axis=0) * size
            ax.plot(t, p[:, 0], color=c, ls=ls, alpha=0.9,
                    lw=2.0 - 0.9 * k / max(len(by_model) - 1, 1), label=model)
        ax.set_xlabel("frame"); ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("x position (px)")
            ax.legend(fontsize=6.5)
    fig.suptitle("Predicted vs true trajectories (seed-averaged)\n"
                 "top: path over the frame, bright band = barrier, markers = hidden frames; "
                 "bottom: x over time, grey span = fully hidden",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir, "trajectories.png")


def fig_error_over_time(labels, by_model, out_dir):
    """Mean error aligned on the moment the target fully reappears.

    Aligning on reappearance (rather than on the frame index) puts every
    video's occlusion on the same x axis, so the spike during occlusion and the
    recovery afterwards are comparable across videos of different durations.
    """
    vf = labels["visible_frac"]
    reap = reappearance_index(vf)
    keep = np.flatnonzero(reap >= 0)
    if keep.size == 0:
        return None
    window = 12
    offsets = np.arange(-window, window + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for model, runs in by_model.items():
        c, m, ls = style(model)
        err = np.mean([r["err"] for r in runs], axis=0)
        curve = np.full(offsets.size, np.nan)
        for j, off in enumerate(offsets):
            idx = reap[keep] + off
            ok = (idx >= 0) & (idx < err.shape[1])
            if ok.sum():
                curve[j] = err[keep[ok], idx[ok]].mean()
        ax.plot(offsets, curve, color=c, marker=m, ls=ls, ms=4, lw=1.6, label=model)
    ax.axvline(0, color="0.4", lw=1.0)
    ax.text(0, ax.get_ylim()[1], " target fully visible again", fontsize=7, va="top",
            color="0.3", rotation=90)
    ax.axhline(SUCCESS_PX, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("frames relative to full reappearance")
    ax.set_ylabel("mean center error (px)")
    ax.set_title(f"Error around reappearance ({keep.size} videos)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "error_over_time.png")


def fig_error_by_category(labels, by_model, out_dir, n_boot=2000):
    """Per-category error with bootstrap intervals: grouped bars."""
    masks = category_masks(labels["visible_frac"])
    models = list(by_model)
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(CATEGORIES))
    for i, model in enumerate(models):
        c, _, _ = style(model)
        means, lo, hi = [], [], []
        for cat in CATEGORIES:
            vals = np.mean([r["per_video"][cat] for r in by_model[model]], axis=0)
            m, l, h, _ = stats.bootstrap_mean_ci(vals, n_boot=n_boot)
            means.append(m); lo.append(m - l); hi.append(h - m)
        ax.bar(x + i * width - 0.4 + width / 2, means, width * 0.92, label=model, color=c,
               yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.9))
    ax.axhline(SUCCESS_PX, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("center error (px)")
    ax.set_title("Error by visibility category (95% bootstrap CI over videos)", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "error_by_category.png")


def make_all(preds_dir="preds", source="generator", n=None, seed=None, out_dir="figures",
             strict_tag=True, n_boot=2000):
    labels, entries, problems = load_all(preds_dir, source=source, strict_tag=strict_tag,
                                         n=n, seed=seed)
    if not entries:
        raise SystemExit(f"no usable prediction files in {preds_dir}/ "
                         f"(run eval.report --make-fake for placeholder data)")
    data = build_runs(labels, entries)
    for r, e in zip(data["runs"], entries):      # trajectory plots need the raw predictions
        r["pred"] = e["pred_xy"]
    by_model = group_by_model(data["runs"])
    written = [
        fig_error_vs_duration(by_model, data, out_dir),
        fig_recovery_vs_duration(by_model, data, out_dir),
        fig_trajectories(labels, by_model, out_dir),
        fig_error_over_time(labels, by_model, out_dir),
        fig_error_by_category(labels, by_model, out_dir, n_boot=n_boot),
    ]
    return [w for w in written if w], problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds-dir", default="preds")
    p.add_argument("--data", default="generator", choices=["generator", "starter"])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--make-fake", action="store_true")
    p.add_argument("--allow-tag-mismatch", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args(argv)
    if a.make_fake:
        from eval.loader import load_labels
        make_fake_predictions(load_labels(a.data, n=a.n, seed=a.data_seed), out_dir=a.preds_dir)
    written, problems = make_all(a.preds_dir, a.data, a.n, a.data_seed, a.out_dir,
                                 strict_tag=not a.allow_tag_mismatch, n_boot=a.n_boot)
    for path, msg in problems:
        print(f"skipped {os.path.basename(path)}: {msg}")
    for w in written:
        print(f"wrote {w}")


if __name__ == "__main__":
    main()
