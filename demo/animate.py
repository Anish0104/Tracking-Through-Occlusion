"""Side-by-side animation: the scene, each model's marker and the true path.

    PYTHONPATH=. python -m demo.animate                        # best + worst cases
    PYTHONPATH=. python -m demo.animate --videos 3 17 42       # specific videos
    PYTHONPATH=. python -m demo.animate --pick worst --n-videos 4 --format mp4

Writes demo/*.gif (or .mp4 with ffmpeg installed). Each panel is one video:
the frame as the models see it, the true path so far, and one marker per
model. The title shows the frame number, the visibility state and the current
error, so a failure is visible as it happens rather than only in a table.

Failure cases are included on purpose (eval/README.md): ``--pick worst``
selects the videos with the largest hidden-frame error, which is where the
models differ and where the interesting behaviour is.
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from eval.figures import style
from eval.loader import load_all, make_fake_predictions
from eval.metrics import SIZE, hidden_durations, per_video_values, pixel_error, category_masks
from eval.report import build_runs, group_by_model

FPS = 6


def pick_videos(labels, by_model, how="mix", n_videos=3, focus=None):
    """Choose which videos to animate.

    ``worst``/``best`` rank by hidden-frame error (averaged over models, or over
    ``focus`` if given); ``longest`` takes the longest occlusions; ``mix`` shows
    the two worst and one typical case, which is the most informative default
    for a demo that is meant to include failures.
    """
    masks = category_masks(labels["visible_frac"])
    models = [focus] if focus and focus in by_model else list(by_model)
    per_model = []
    for m in models:
        runs = by_model[m]
        err = np.mean([r["err"] for r in runs], axis=0)
        per_model.append(per_video_values(err, masks["hidden"]))
    with warnings.catch_warnings():
        # videos with no hidden frame are all-nan here, which is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        score = np.nanmean(np.vstack(per_model), axis=0)
    dur = hidden_durations(labels["visible_frac"])
    occluded = np.flatnonzero(dur > 0)
    if occluded.size == 0:
        return list(range(min(n_videos, labels["n"])))
    order = occluded[np.argsort(-np.nan_to_num(score[occluded], nan=-np.inf))]
    if how == "worst":
        return [int(v) for v in order[:n_videos]]
    if how == "best":
        return [int(v) for v in order[::-1][:n_videos]]
    if how == "longest":
        return [int(v) for v in occluded[np.argsort(-dur[occluded])][:n_videos]]
    # mix: worst two (failures) plus a median case (typical behaviour)
    picks = [int(v) for v in order[:max(n_videos - 1, 1)]]
    median = order[len(order) // 2]
    if median not in picks:
        picks.append(int(median))
    return [int(v) for v in picks[:n_videos]]


def animate(labels, by_model, videos, out_path, fps=FPS, trail=True):
    """Write one animation covering ``videos`` as side-by-side panels."""
    frames_all = labels["frames"]
    xy = labels["xy"]
    vf = labels["visible_frac"]
    n_panels = len(videos)
    T = frames_all.shape[1]

    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 4.3), squeeze=False)
    axes = axes[0]
    preds = {m: np.mean([r["pred"] for r in runs], axis=0) for m, runs in by_model.items()}
    errs = {m: np.mean([r["err"] for r in runs], axis=0) for m, runs in by_model.items()}

    artists = []
    for ax, vid in zip(axes, videos):
        ax.set_xlim(0, SIZE); ax.set_ylim(SIZE, 0)
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(frames_all[vid][0], cmap="gray", vmin=0, vmax=1,
                       extent=(0, SIZE, SIZE, 0), interpolation="nearest")
        true_line, = ax.plot([], [], color="#00ff88", lw=1.4, alpha=0.75,
                             label="true path", zorder=6)
        true_dot, = ax.plot([], [], color="#00ff88", marker="o", ms=13, mfc="none", mew=2.4,
                            ls="none", label="true", zorder=20)
        model_dots, model_trails = {}, {}
        for model in by_model:
            c, m, ls = style(model)
            model_dots[model], = ax.plot([], [], color=c, marker=m, ms=7, ls="none",
                                         label=model, mew=1.8, alpha=0.95, zorder=10)
            model_trails[model], = ax.plot([], [], color=c, lw=1.0, alpha=0.55, ls=ls)
        title = ax.set_title("", fontsize=8.5)
        artists.append({"ax": ax, "vid": vid, "im": im, "true_line": true_line,
                        "true_dot": true_dot, "dots": model_dots, "trails": model_trails,
                        "title": title})
    axes[0].legend(fontsize=6.5, loc="upper left", framealpha=0.85)

    def draw(t):
        changed = []
        for a in artists:
            vid = a["vid"]
            a["im"].set_data(frames_all[vid][t])
            p = xy[vid] * SIZE
            if trail:
                a["true_line"].set_data(p[:t + 1, 0], p[:t + 1, 1])
            a["true_dot"].set_data([p[t, 0]], [p[t, 1]])
            v = float(vf[vid][t])
            state = "HIDDEN" if v <= 0 else ("visible" if v >= 1 else f"partial {v:.0%}")
            bits = []
            for model in by_model:
                q = preds[model][vid] * SIZE
                a["dots"][model].set_data([q[t, 0]], [q[t, 1]])
                if trail:
                    a["trails"][model].set_data(q[:t + 1, 0], q[:t + 1, 1])
                bits.append(f"{model} {errs[model][vid][t]:.1f}")
            a["title"].set_text(f"video {vid}  t={t:02d}  {state}\n"
                                + "  ".join(bits) + "  (px error)")
            changed += [a["im"], a["true_line"], a["true_dot"], a["title"]]
            changed += list(a["dots"].values()) + list(a["trails"].values())
        return changed

    fig.suptitle("Tracking through occlusion: model markers vs the true position", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    anim = animation.FuncAnimation(fig, draw, frames=T, interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith(".mp4"):
        try:
            anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        except Exception as exc:                 # ffmpeg missing: GIF always works
            out_path = out_path[:-4] + ".gif"
            print(f"mp4 writer unavailable ({exc.__class__.__name__}), writing {out_path}")
            anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds-dir", default="preds")
    p.add_argument("--data", default="generator", choices=["generator", "starter"])
    p.add_argument("--n", type=int, default=None, help="videos to load (default: full bank)")
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--videos", type=int, nargs="+", default=None, help="explicit video indices")
    p.add_argument("--pick", default="mix", choices=["mix", "worst", "best", "longest"],
                   help="how to choose videos when --videos is not given")
    p.add_argument("--n-videos", type=int, default=3)
    p.add_argument("--focus", default=None, help="rank worst/best by this model only")
    p.add_argument("--out", default=None, help="output path (.gif or .mp4)")
    p.add_argument("--format", default="gif", choices=["gif", "mp4"])
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--make-fake", action="store_true")
    p.add_argument("--allow-tag-mismatch", action="store_true")
    a = p.parse_args(argv)

    if a.make_fake:
        from eval.loader import load_labels
        make_fake_predictions(load_labels(a.data, n=a.n, seed=a.data_seed), out_dir=a.preds_dir)

    labels, entries, problems = load_all(a.preds_dir, source=a.data,
                                         strict_tag=not a.allow_tag_mismatch,
                                         n=a.n, seed=a.data_seed)
    for path, msg in problems:
        print(f"skipped {os.path.basename(path)}: {msg}")
    if not entries:
        raise SystemExit(f"no usable prediction files in {a.preds_dir}/")
    data = build_runs(labels, entries)
    for r, e in zip(data["runs"], entries):
        r["pred"] = e["pred_xy"]
    by_model = group_by_model(data["runs"])

    videos = a.videos if a.videos else pick_videos(labels, by_model, a.pick, a.n_videos, a.focus)
    bad = [v for v in videos if not 0 <= v < labels["n"]]
    if bad:
        raise SystemExit(f"video index out of range for {labels['n']} videos: {bad}")
    out = a.out or os.path.join("demo", f"tracking_{a.pick}.{a.format}")
    written = animate(labels, by_model, videos, out, fps=a.fps)
    print(f"wrote {written}  (videos {list(videos)})")


if __name__ == "__main__":
    main()
