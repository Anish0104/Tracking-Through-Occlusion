"""The project's evaluation report: reads preds/, prints every table.

    PYTHONPATH=. python -m eval.report                      # official test bank
    PYTHONPATH=. python -m eval.report --data starter --n 200 --make-fake
    PYTHONPATH=. python -m eval.report --json results.json   # also save machine-readable

What it covers (eval/README.md, baselines/HANDOFF.md "What evaluation should compute"):

1. error by visibility category, and success rate at 5 px
2. recovery delay and never-recovered fraction, plus a strict-threshold
   sensitivity check (the 5 px numbers saturate for the better models)
3. everything grouped by hidden duration, with 12 flagged as extrapolation
4. per-seed results for the learned models with mean and spread over seeds
5. paired per-video comparisons with bootstrap confidence intervals
6. a compute table (parameters, training and inference time)
7. spiking activity for the SNN

Models are aggregated over seeds by averaging each seed's per-video error
first, so a model with three seeds is not given extra weight in comparisons.
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np

from eval import stats
from eval.loader import SNN_NEURONS, load_all, make_fake_predictions, model_sort_key
from eval.metrics import (CATEGORIES, RECOVERY_FRAMES, SUCCESS_PX, category_masks,
                          has_occlusion, has_reappearance, hidden_durations, per_video_mean,
                          per_video_values, pixel_error, recovery_delay, summarize)

#: Durations the models were trained on; 12 is never seen in training.
TRAIN_DURATIONS = (0, 2, 4, 8)
EXTRAPOLATION_DURATION = 12
#: Stricter recovery threshold for the sensitivity check (handoff note).
STRICT_RECOVERY_PX = 1.0


def _fmt(x, nd=3, width=0):
    s = "   -  " if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"
    return f"{s:>{width}}" if width else s


def _pct(x, nd=1):
    return "  -  " if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


# --------------------------------------------------------------------------- #
# per-model data assembled once, then reused by every table
# --------------------------------------------------------------------------- #

def build_runs(labels, entries):
    """Per-file metrics plus the per-video arrays the statistics need."""
    masks = category_masks(labels["visible_frac"])
    dur = hidden_durations(labels["visible_frac"])
    occ = has_occlusion(labels["visible_frac"])
    reap = has_reappearance(labels["visible_frac"])
    runs = []
    for e in entries:
        err = pixel_error(e["pred_xy"], labels["xy"])
        delay, recovered = recovery_delay(err, labels["visible_frac"])
        s_delay, s_recovered = recovery_delay(err, labels["visible_frac"],
                                              threshold=STRICT_RECOVERY_PX)
        runs.append({
            "model": e["model"], "seed": e["seed"], "path": e["path"],
            "n_params": e["n_params"], "spikes": e["spikes"],
            "summary": summarize(e["pred_xy"], labels["xy"], labels["visible_frac"],
                                 e["spikes"], SNN_NEURONS if e["spikes"] is not None else None),
            "err": err,
            # per-video error per category: the unit of every paired comparison
            "per_video": {c: per_video_values(err, masks[c]) for c in CATEGORIES},
            "delay": delay, "recovered": recovered,
            "strict_delay": s_delay, "strict_recovered": s_recovered,
        })
    return {"runs": runs, "masks": masks, "durations": dur, "occlusion": occ,
            "reappearance": reap}


def group_by_model(runs):
    """{model: [run, ...]} preserving the table order."""
    by = defaultdict(list)
    for r in runs:
        by[r["model"]].append(r)
    return {m: sorted(by[m], key=lambda r: r["seed"])
            for m in sorted(by, key=model_sort_key)}


def seed_averaged_per_video(model_runs, category):
    """Mean per-video error across a model's seeds. -> [N]

    Averaging seeds before comparing models keeps each *model* as one unit, so
    a 3-seed model does not dominate a 1-seed baseline in the paired tests.
    """
    stack = np.vstack([r["per_video"][category] for r in model_runs])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # a video with no frame of this category is all-nan: that is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def table_overall(by_model):
    """Error and success per category, per seed, with the mean +/- sd over seeds."""
    out = [f"{'model':16s} {'seed':>5s} " + " ".join(f"{c:>9s}" for c in CATEGORIES)
           + f" {'<=5px':>8s} {'recov':>7s} {'never':>7s}"]
    for model, runs in by_model.items():
        for r in runs:
            cats = r["summary"]["categories"]
            rec = r["summary"]["recovery"]
            out.append(
                f"{model:16s} {r['seed']:5d} "
                + " ".join(_fmt(cats[c]["err_px"], 3, 9) for c in CATEGORIES)
                + f" {_pct(cats['overall']['success']):>8s}"
                + f" {_fmt(rec['delay_mean'], 2, 7)} {_pct(rec['never_recovered']):>7s}")
        if len(runs) > 1:                      # spread over seeds (CONTRACT.md section 4)
            line = f"{'  mean +/- sd':16s} {len(runs):5d} "
            for c in CATEGORIES:
                m, sd, _ = stats.mean_sd([r["summary"]["categories"][c]["err_px"] for r in runs])
                line += f"{(_fmt(m, 2) + '+-' + _fmt(sd, 2)):>9s} "
            out.append(line.rstrip())
    return "\n".join(out)


def duration_breakdown(by_model, data):
    """{model: {duration: {err_px, hidden_px, success, delay, never}}}"""
    durations, masks = data["durations"], data["masks"]
    reap = data["reappearance"]
    present = sorted(set(durations.tolist()))
    out = {}
    for model, runs in by_model.items():
        per_dur = {}
        for d in present:
            sel = durations == d
            vals = defaultdict(list)
            for r in runs:
                err = r["err"][sel]
                ov, _ = per_video_mean(err, masks["overall"][sel])
                hd, _ = per_video_mean(err, masks["hidden"][sel])
                sc, _ = per_video_mean((err <= SUCCESS_PX).astype(float), masks["overall"][sel])
                vals["err_px"].append(ov)
                vals["hidden_px"].append(hd)
                vals["success"].append(sc)
                rec, dly = r["recovered"][sel], r["delay"][sel]
                vals["delay"].append(float(np.nanmean(dly[rec])) if rec.any() else np.nan)
                o = reap[sel]
                vals["never"].append(float((~rec)[o].mean()) if o.any() else np.nan)
                srec = r["strict_recovered"][sel]
                sdly = r["strict_delay"][sel]
                vals["strict_delay"].append(float(np.nanmean(sdly[srec])) if srec.any() else np.nan)
                vals["strict_never"].append(float((~srec)[o].mean()) if o.any() else np.nan)
            per_dur[int(d)] = {k: stats.mean_sd(v)[:2] for k, v in vals.items()}
            per_dur[int(d)]["n_videos"] = int(sel.sum())
        out[model] = per_dur
    return out


def format_duration_tables(breakdown, metric, title, nd=3, pct=False):
    """One table: models as rows, hidden durations as columns."""
    durs = sorted(next(iter(breakdown.values())).keys())
    head = f"{'model':16s} " + " ".join(
        f"{('d=' + str(d)) + ('*' if d == EXTRAPOLATION_DURATION else ''):>13s}" for d in durs)
    lines = [title, head]
    for model, per_dur in breakdown.items():
        row = f"{model:16s} "
        for d in durs:
            m, sd = per_dur[d][metric]
            cell = _pct(m) if pct else _fmt(m, nd)
            if np.isfinite(sd) and sd > 0:
                cell += f"+-{_fmt(sd, nd)}"
            row += f"{cell:>13s} "
        lines.append(row.rstrip())
    return "\n".join(lines)


def comparison_table(by_model, category="hidden", reference=None, n_boot=stats.N_BOOT):
    """Paired per-video comparisons against a reference model, with bootstrap CIs.

    Negative difference = the model has lower error than the reference. The
    interval is over per-video differences on the same videos, which is what
    makes sub-pixel gaps resolvable.
    """
    models = list(by_model)
    if reference is None:
        for cand in ("last_pos", "kalman", "gru"):
            if cand in by_model:
                reference = cand
                break
        else:
            reference = models[-1]
    ref_vals = seed_averaged_per_video(by_model[reference], category)
    lines = [f"paired per-video difference in {category}-frame error vs {reference} "
             f"(95% bootstrap CI, {n_boot} resamples)",
             f"{'model':16s} {'diff px':>9s} {'95% CI':>20s} {'better':>8s} {'n':>6s} {'resolved':>9s}"]
    rows = {}
    for model in models:
        if model == reference:
            continue
        r = stats.paired_diff_ci(seed_averaged_per_video(by_model[model], category),
                                 ref_vals, n_boot=n_boot)
        rows[model] = r
        ci = f"[{_fmt(r['low'], 3)}, {_fmt(r['high'], 3)}]"
        lines.append(f"{model:16s} {_fmt(r['mean_diff'], 3, 9)} {ci:>20s} "
                     f"{_pct(r['frac_a_better']):>8s} {r['n_pairs']:6d} "
                     f"{('yes' if r['significant'] else 'no'):>9s}")
    return "\n".join(lines), {"reference": reference, "category": category, "rows": rows}


def absolute_ci_table(by_model, category="hidden", n_boot=stats.N_BOOT):
    """Bootstrap CI for each model's own mean error (not a difference)."""
    lines = [f"{category}-frame error with 95% bootstrap CI over videos",
             f"{'model':16s} {'mean px':>9s} {'95% CI':>20s} {'videos':>7s}"]
    rows = {}
    for model, runs in by_model.items():
        m, lo, hi, n = stats.bootstrap_mean_ci(seed_averaged_per_video(runs, category),
                                               n_boot=n_boot)
        rows[model] = {"mean": m, "low": lo, "high": hi, "n": n}
        ci = f"[{_fmt(lo, 3)}, {_fmt(hi, 3)}]"
        lines.append(f"{model:16s} {_fmt(m, 3, 9)} {ci:>20s} {n:7d}")
    return "\n".join(lines), rows


def recovery_table(by_model):
    """Recovery at the contract threshold and at a stricter one."""
    lines = [f"recovery: error <= threshold for {RECOVERY_FRAMES} frames in a row after full reappearance",
             f"{'model':16s} {'delay@5px':>10s} {'never@5px':>10s} "
             f"{('delay@' + str(STRICT_RECOVERY_PX) + 'px'):>12s} {('never@' + str(STRICT_RECOVERY_PX) + 'px'):>12s}"]
    rows = {}
    for model, runs in by_model.items():
        d = stats.mean_sd([r["summary"]["recovery"]["delay_mean"] for r in runs])[0]
        nv = stats.mean_sd([r["summary"]["recovery"]["never_recovered"] for r in runs])[0]
        sd_ = [float(np.nanmean(r["strict_delay"][r["strict_recovered"]]))
               if r["strict_recovered"].any() else np.nan for r in runs]
        strict_delay = stats.mean_sd(sd_)[0]
        strict_never = stats.mean_sd(
            [float((~r["strict_recovered"])[r["reap"]].mean()) if r["reap"].any() else np.nan
             for r in runs])[0]
        rows[model] = {"delay": d, "never": nv,
                       "strict_delay": strict_delay, "strict_never": strict_never}
        lines.append(f"{model:16s} {_fmt(d, 2, 10)} {_pct(nv):>10s} "
                     f"{_fmt(strict_delay, 2, 12)} {_pct(strict_never):>12s}")
    return "\n".join(lines), rows


def motion_table(labels, by_model, category="hidden", tol=1e-6):
    """Error split by motion type, inferred from the labels.

    The handoff notes Kalman degrades badly on smoothly varying speed while the
    GRU does not, so the split matters for reading the comparison. Constant
    speed means a constant per-frame step in x (the generator's "constant"
    mode); anything else is "smooth".
    """
    x = np.asarray(labels["xy"], np.float64)[..., 0]
    step = np.diff(x, axis=1)
    constant = (step.max(axis=1) - step.min(axis=1)) < tol
    lines = [f"{category}-frame error by motion type "
             f"(constant {int(constant.sum())} videos, smooth {int((~constant).sum())})",
             f"{'model':16s} {'constant':>10s} {'smooth':>10s} {'ratio':>8s}"]
    rows = {}
    for model, runs in by_model.items():
        v = seed_averaged_per_video(runs, category)
        with np.errstate(invalid="ignore"):
            c = float(np.nanmean(v[constant])) if constant.any() else np.nan
            s = float(np.nanmean(v[~constant])) if (~constant).any() else np.nan
        ratio = s / c if (np.isfinite(c) and c > 0) else np.nan
        rows[model] = {"constant": c, "smooth": s, "ratio": ratio}
        lines.append(f"{model:16s} {_fmt(c, 3, 10)} {_fmt(s, 3, 10)} {_fmt(ratio, 2, 8)}")
    return "\n".join(lines), rows


def spike_table(by_model):
    """SNN spiking activity: the energy-proxy column of the compute table."""
    rows = {}
    lines = [f"spiking activity ({SNN_NEURONS:,} spiking neurons)",
             f"{'model':16s} {'seed':>5s} {'spikes/frame':>13s} {'spikes/neuron/frame':>20s}"]
    any_spikes = False
    for model, runs in by_model.items():
        for r in runs:
            s = r["summary"]
            if "spikes_per_frame" not in s:
                continue
            any_spikes = True
            rows.setdefault(model, {})[r["seed"]] = {
                "per_frame": s["spikes_per_frame"],
                "per_neuron_per_frame": s["spikes_per_neuron_per_frame"]}
            lines.append(f"{model:16s} {r['seed']:5d} {s['spikes_per_frame']:13.1f} "
                         f"{s['spikes_per_neuron_per_frame']:20.5f}")
    return ("\n".join(lines) if any_spikes else
            "spiking activity: no prediction file carried a spikes array"), rows


def compute_table(by_model, timings_path="eval/compute.json"):
    """Parameters, training time and inference time.

    Parameter counts come from the prediction files. Timings cannot: they are
    measured on the machine that trained each model, so they are read from
    ``eval/compute.json`` (written by whoever ran the training) and shown as
    missing until that file is filled in.
    """
    timings = {}
    note = f"timings from {timings_path}"
    if os.path.exists(timings_path):
        with open(timings_path) as f:
            timings = json.load(f)
    else:
        note = (f"{timings_path} not found: fill it in with train_s / infer_ms_per_video "
                f"per model, measured on one machine")
    lines = ["compute", f"{'model':16s} {'params':>12s} {'train (s)':>11s} {'infer (ms/video)':>17s}"]
    rows = {}
    for model, runs in by_model.items():
        params = next((r["n_params"] for r in runs if r["n_params"]), None)
        t = timings.get(model, {})
        rows[model] = {"n_params": params, "train_s": t.get("train_s"),
                       "infer_ms_per_video": t.get("infer_ms_per_video")}
        lines.append(f"{model:16s} {(f'{params:,}' if params else '    -   '):>12s} "
                     f"{_fmt(t.get('train_s'), 1, 11)} {_fmt(t.get('infer_ms_per_video'), 2, 17)}")
    lines.append(f"  ({note})")
    return "\n".join(lines), rows


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run(preds_dir="preds", source="generator", n=None, seed=None, strict_tag=True,
        n_boot=stats.N_BOOT, timings_path="eval/compute.json"):
    """Build every table. Returns ``(text, results_dict)``."""
    labels, entries, problems = load_all(preds_dir, source=source, strict_tag=strict_tag,
                                         n=n, seed=seed)
    if not entries:
        detail = "".join(f"\n  {os.path.basename(p)}: {m}" for p, m in problems)
        raise SystemExit(f"no usable prediction files in {preds_dir}/{detail}\n"
                         f"Generate real ones (see baselines/HANDOFF.md) or pass "
                         f"--make-fake to build the pipeline on placeholder data.")
    data = build_runs(labels, entries)
    for r in data["runs"]:                       # recovery_table needs these masks
        r["occ"] = data["occlusion"]
        r["reap"] = data["reappearance"]
    by_model = group_by_model(data["runs"])

    out = [f"Tracking Through Occlusion - evaluation",
           f"data: {labels['tag']} ({labels['n']} videos, {labels['xy'].shape[1]} frames, "
           f"source {labels['source']})",
           f"files: {len(entries)} prediction file(s) in {preds_dir}/",
           f"durations present: {sorted(set(data['durations'].tolist()))} "
           f"(* = {EXTRAPOLATION_DURATION} is extrapolation, never trained on)",
           f"aggregation: mean per video, then over videos; success and recovery at {SUCCESS_PX:g} px"]
    if problems:
        out.append("skipped files:")
        out += [f"  {os.path.basename(p)}: {m}" for p, m in problems]
    warn = [e["warning"] for e in entries if e.get("warning")]
    if warn:
        out.append("warnings:")
        out += [f"  {w}" for w in warn]

    results = {"data_tag": labels["tag"], "n_videos": labels["n"], "source": labels["source"],
               "n_files": len(entries), "skipped": [{"path": p, "error": m} for p, m in problems]}

    def section(title, body):
        out.extend(["", "=" * 78, title, "=" * 78, body])

    section("1. error (px) by visibility category, and success at 5 px", table_overall(by_model))

    breakdown = duration_breakdown(by_model, data)
    results["by_duration"] = {m: {str(d): {k: (list(v) if isinstance(v, tuple) else v)
                                           for k, v in dv.items()}
                                  for d, dv in per.items()} for m, per in breakdown.items()}
    section("2. by hidden duration",
            "\n\n".join([
                format_duration_tables(breakdown, "err_px", "all-frame error (px)"),
                format_duration_tables(breakdown, "hidden_px", "hidden-frame error (px)"),
                format_duration_tables(breakdown, "success", f"success rate (<= {SUCCESS_PX:g} px)", pct=True),
                format_duration_tables(breakdown, "delay", "recovery delay (frames, recovered videos)", nd=2),
                format_duration_tables(breakdown, "never", "never recovered", pct=True),
            ]))

    rec_text, rec_rows = recovery_table(by_model)
    results["recovery"] = rec_rows
    section("3. recovery", rec_text)

    abs_text, abs_rows = absolute_ci_table(by_model, "hidden", n_boot)
    abs_all_text, abs_all_rows = absolute_ci_table(by_model, "overall", n_boot)
    results["ci_hidden"], results["ci_overall"] = abs_rows, abs_all_rows
    section("4. bootstrap confidence intervals", abs_text + "\n\n" + abs_all_text)

    cmp_parts, cmp_results = [], {}
    for cat in ("hidden", "overall"):
        text, meta = comparison_table(by_model, cat, n_boot=n_boot)
        cmp_parts.append(text)
        cmp_results[cat] = meta
    if "snn" in by_model and "snn_reset" in by_model:
        text, meta = comparison_table({k: v for k, v in by_model.items()
                                       if k in ("snn", "snn_reset")},
                                      "hidden", reference="snn_reset", n_boot=n_boot)
        cmp_parts.append("state-reset ablation (does the recurrent state help?)\n" + text)
        cmp_results["ablation"] = meta
    results["comparisons"] = cmp_results
    section("5. paired comparisons", "\n\n".join(cmp_parts))

    mot_text, mot_rows = motion_table(labels, by_model)
    results["motion"] = mot_rows
    section("6. motion type", mot_text)

    spk_text, spk_rows = spike_table(by_model)
    results["spikes"] = spk_rows
    section("7. spiking activity", spk_text)

    comp_text, comp_rows = compute_table(by_model, timings_path)
    results["compute"] = comp_rows
    section("8. compute", comp_text)

    results["overall"] = {
        m: {"per_seed": [{"seed": r["seed"],
                          "categories": r["summary"]["categories"],
                          "recovery": r["summary"]["recovery"]} for r in runs]}
        for m, runs in by_model.items()}
    return "\n".join(out), results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds-dir", default="preds")
    p.add_argument("--data", default="generator", choices=["generator", "starter"],
                   help="label source; generator = official test bank")
    p.add_argument("--n", type=int, default=None, help="videos (default: the full bank)")
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--make-fake", action="store_true",
                   help="write placeholder predictions first (development only)")
    p.add_argument("--allow-tag-mismatch", action="store_true",
                   help="score files whose data_tag differs from the label source")
    p.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    p.add_argument("--timings", default="eval/compute.json")
    p.add_argument("--json", default=None, help="also write the results as JSON")
    p.add_argument("--out", default=None, help="also write the printed report to a text file")
    a = p.parse_args(argv)

    if a.make_fake:
        from eval.loader import load_labels
        labels = load_labels(a.data, n=a.n, seed=a.data_seed)
        written = make_fake_predictions(labels, out_dir=a.preds_dir)
        print(f"wrote {len(written)} placeholder prediction files to {a.preds_dir}/ "
              f"(development only, not results)\n")

    text, results = run(a.preds_dir, a.data, a.n, a.data_seed,
                        strict_tag=not a.allow_tag_mismatch, n_boot=a.n_boot,
                        timings_path=a.timings)
    print(text)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nwrote {a.json}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {a.out}")
    return results


if __name__ == "__main__":
    main()
