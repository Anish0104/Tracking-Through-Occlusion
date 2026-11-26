"""Validation-only analysis of the final CNN-GRU seeds against the classical baselines.

    PYTHONPATH=. python -m baselines.validate_final            # writes <exp>/validation/

Development analysis inside baselines/, not the project evaluation (eval/). It reads the
cached validation split (data.generator, seed 200, 1,000 videos) and never generates or
reads the test bank (seed 300). Models receive frames only; labels are used only to score.

Metric definitions (pixels = normalized difference x 64):
  category means      per-frame error averaged within each video, then over the videos that
                      have such frames (dev_metrics.summarize; eval/README.md wording)
  median, RMSE        over all 40,000 frames
  within 5 px         per video, then over videos (CONTRACT.md success threshold)
  duration d          videos with exactly d fully hidden frames (generator guarantees this)
  recovery            CONTRACT.md: error <= 5 px for 3 frames in a row after full
                      reappearance. Full reappearance t0 = first frame after the last fully
                      hidden frame with visible_frac == 1. Delay = frames from t0 to the start
                      of the first such 3-frame run (0 = recovered at once); "never" if no run
                      fits before the end of the video. Stricter 1 px / 2 px thresholds are
                      reported as a sensitivity check only.
  motion              "constant" if the true per-frame x step is constant (std < 1e-3 px),
                      else "smooth" (the generator's sinusoidal speed modulation)
"""

import argparse
import csv
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from baselines import data_cache
from baselines.detector import detect
from baselines.dev_metrics import TEST_SEED, summarize
from baselines.kalman import KalmanConfig, config_dict, load_config, track
from common.loss import balanced_mse

SIZE = 64
SUCCESS_PX = 5.0
DURATIONS = (0, 2, 4, 8)
VAL_N, VAL_SEED = 1000, 200
CATS = ("overall", "visible", "partial", "hidden")

# tune_kalman.GRID (the agreed protocol grid) and an extension that brackets the edges it hit.
from baselines.tune_kalman import FIXED, GRID  # noqa: E402

EXTENDED_GRID = {
    "accel_px": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
    "meas_px": [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4],
    "partial_px": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0],
    "min_pixels": [1, 2, 4],          # 1 is the hard lower bound (KalmanConfig requires >= 1)
}


# ---------------------------------------------------------------- metrics

def pixel_error(pred, xy):
    return np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(xy, np.float64), axis=-1) * SIZE


def bmse(pred, xy, vis):
    return float(balanced_mse(torch.tensor(np.asarray(pred, np.float32)),
                              torch.tensor(np.asarray(xy, np.float32)),
                              torch.tensor(np.asarray(vis, np.float32))))


def hidden_spans(vis):
    """Per video: (first hidden frame, last hidden frame) or None; checks contiguity."""
    out = []
    for v in vis:
        h = np.flatnonzero(v == 0)
        if len(h) == 0:
            out.append(None)
            continue
        assert np.all(np.diff(h) == 1), "hidden frames are not contiguous"
        out.append((int(h[0]), int(h[-1])))
    return out


def recovery(err, vis, spans, thr=SUCCESS_PX, run=3):
    """Per-video recovery delay in frames (np.inf = never, nan = no occlusion)."""
    T = err.shape[1]
    delay = np.full(len(err), np.nan)
    for i, sp in enumerate(spans):
        if sp is None:
            continue
        after = np.flatnonzero(vis[i, sp[1] + 1:] == 1)
        if len(after) == 0:
            delay[i] = np.inf
            continue
        t0 = sp[1] + 1 + int(after[0])
        ok = err[i] <= thr
        delay[i] = np.inf
        for t in range(t0, T - run + 1):
            if ok[t:t + run].all():
                delay[i] = t - t0
                break
    return delay


def _rec_summary(d):
    d = d[~np.isnan(d)]
    fin = d[np.isfinite(d)]
    return {"videos": int(len(d)), "never_frac": float(np.mean(~np.isfinite(d))) if len(d) else float("nan"),
            "mean_delay": float(fin.mean()) if len(fin) else float("nan"),
            "max_delay": float(fin.max()) if len(fin) else float("nan"),
            "immediate_frac": float(np.mean(d == 0)) if len(d) else float("nan")}


def metrics(pred, xy, vis, durations, spans, motion):
    err = pixel_error(pred, xy)
    s = summarize(pred, xy, vis)
    out = {c: s[c]["err_px"] for c in CATS}
    out["median_px"] = float(np.median(err))
    out["rmse_px"] = float(np.sqrt(np.mean(err ** 2)))
    out["within5"] = s["overall"]["success"]
    out["hidden_within5"] = s["hidden"]["success"]
    out["max_px"] = float(err.max())
    out["val_balanced_mse"] = bmse(pred, xy, vis)

    out["by_duration"] = {}
    for d in DURATIONS:
        m = durations == d
        sd = summarize(pred[m], xy[m], vis[m])
        out["by_duration"][d] = {**{c: sd[c]["err_px"] for c in CATS},
                                 "within5": sd["overall"]["success"], "videos": int(m.sum()),
                                 "last_hidden_px": float("nan")}
    # position within the occlusion: k-th fully hidden frame (1-based) for each duration,
    # plus an event-aligned window from 4 frames before the first hidden frame to 8 after the last
    out["by_hidden_position"], out["aligned"] = {}, {}
    for d in DURATIONS[1:]:
        idx = np.flatnonzero(durations == d)
        pos = np.array([[err[i, spans[i][0] + k] for k in range(d)] for i in idx])
        out["by_hidden_position"][d] = pos.mean(0).tolist()
        out["by_duration"][d]["last_hidden_px"] = float(pos[:, -1].mean())
        pre = np.array([[err[i, spans[i][0] + o] for o in range(-4, 0)] for i in idx])
        post = np.array([[err[i, spans[i][1] + o] if spans[i][1] + o < err.shape[1] else np.nan
                          for o in range(1, 9)] for i in idx])
        out["aligned"][d] = {"pre_-4..-1": pre.mean(0).tolist(), "hidden_1..d": pos.mean(0).tolist(),
                             "post_+1..+8": np.nanmean(post, 0).tolist()}

    out["recovery"] = {}
    for thr in (SUCCESS_PX, 2.0, 1.0):
        dl = recovery(err, vis, spans, thr)
        key = f"{thr:g}px"
        out["recovery"][key] = {"all_occluded": _rec_summary(dl),
                                **{d: _rec_summary(dl[durations == d]) for d in DURATIONS[1:]}}

    out["by_motion"] = {}
    for mname in ("constant", "smooth"):
        m = motion == mname
        sm = summarize(pred[m], xy[m], vis[m])
        out["by_motion"][mname] = {**{c: sm[c]["err_px"] for c in CATS}, "videos": int(m.sum())}
    return out, err


def along_track_hidden(pred, xy, vis):
    """Signed error along the direction of travel on fully hidden frames (px; + = ahead)."""
    direction = np.sign(xy[:, -1, 0] - xy[:, 0, 0])[:, None]
    ex = (np.asarray(pred[..., 0], np.float64) - xy[..., 0]) * SIZE * direction
    ey = (np.asarray(pred[..., 1], np.float64) - xy[..., 1]) * SIZE
    h = vis == 0
    return ex, ey, h


# ---------------------------------------------------------------- Kalman tuning

def kalman_grid(det, xy, vis, grid):
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = KalmanConfig(**dict(zip(keys, values)), **FIXED)
        pred = track(*det, cfg)
        s = summarize(pred, xy, vis)
        rows.append({"cfg": cfg, "mean_px": s["overall"]["err_px"], "bmse": bmse(pred, xy, vis),
                     **{c: s[c]["err_px"] for c in CATS[1:]}})
    return rows


def edges(cfg, grid):
    out = []
    for k, vals in grid.items():
        v = getattr(cfg, k)
        if v == min(vals) and not (k == "min_pixels" and v == 1):
            out.append(f"{k}={v} at lower edge")
        elif v == max(vals):
            out.append(f"{k}={v} at upper edge")
    return out


def short(cfg):
    return f"accel={cfg.accel_px} meas={cfg.meas_px} partial={cfg.partial_px} min_px={cfg.min_pixels}"


# ---------------------------------------------------------------- bootstrap

def boot_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_video(err, mask):
    cnt = mask.sum(1)
    v = np.where(cnt > 0, (err * mask).sum(1) / np.maximum(cnt, 1), np.nan)
    return v


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="checkpoints/final_gru_generator_v1")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default=None, help="output directory (default <exp>/validation)")
    a = p.parse_args(argv)
    assert VAL_SEED != TEST_SEED
    exp = Path(a.exp)
    out = Path(a.out) if a.out else exp / "validation"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    from data import generator as gen
    from baselines.train_gru import load_checkpoint
    from baselines.gru import predict as gru_predict
    from baselines.last_pos import predict as lp_predict

    # ---- Step 1: integrity
    fp = data_cache.source_fingerprint(gen)
    expj = json.loads((exp / "experiment.json").read_text())
    n, seed, durs = gen.SPLITS["val"]
    assert (n, seed) == (VAL_N, VAL_SEED)
    val_dir = exp / "data_cache" / data_cache.split_name(gen, n, seed)
    val = data_cache.open_split(val_dir, verify=True)          # SHA-256 of every array file
    manifest = json.loads((val_dir / "manifest.json").read_text())
    frames = np.asarray(val["frames"][:], np.float32)
    xy = np.asarray(val["xy"], np.float32)
    vis = np.asarray(val["visible_frac"], np.float32)
    integrity = {"generator_module": gen.__name__, "generator_fingerprint": fp,
                 "experiment_fingerprint": expj["data"]["fingerprint"],
                 "val_cache": str(val_dir), "val_cache_checksums": "verified",
                 "val_array_sha256": {k: manifest["arrays"][k]["sha256"] for k in ("frames", "xy", "visible_frac")},
                 "commit_trained": expj["commit"], "checkpoints": {}}
    assert fp == expj["data"]["fingerprint"], "generator changed since training"

    durations = (vis == 0).sum(1)
    spans = hidden_spans(vis)
    assert set(np.unique(durations)) <= set(DURATIONS), np.unique(durations)
    steps = np.diff(xy[..., 0].astype(np.float64) * SIZE, axis=1)
    motion = np.where(steps.std(1) < 1e-3, "constant", "smooth")

    nets = {}
    for s in (0, 1, 2):
        ck = exp / f"gru_seed{s}.pt"
        net, payload = load_checkpoint(ck, a.device)
        c = payload["config"]
        summ = json.loads((exp / f"gru_seed{s}.json").read_text())
        checks = {"loads": True, "seed": c["seed"] == s, "fingerprint": c.get("data_fingerprint") == fp,
                  "val_split": (c["n_val"], c["val_seed"]) == (VAL_N, VAL_SEED),
                  "data": c["data"] == "generator", "n_params": payload["n_params"] == 1067064}
        assert all(checks.values()), (s, checks)
        integrity["checkpoints"][s] = {"path": str(ck), "sha256": data_cache._sha256(ck), **checks,
                                       "best_epoch": summ["best_epoch"],
                                       "best_val_loss": summ["best_val_loss"],
                                       "epochs_run": summ["epochs_run"], "history": summ["history"]}
        nets[s] = net

    # ---- Step 2: Kalman tuning (validation only)
    det = detect(frames)
    t0 = time.time()
    proto = kalman_grid(det, xy, vis, GRID)
    ext = kalman_grid(det, xy, vis, EXTENDED_GRID)
    t_tune = time.time() - t0
    picks = {
        "protocol_grid_meanpx": (min(proto, key=lambda r: r["mean_px"]), GRID),
        "protocol_grid_bmse": (min(proto, key=lambda r: r["bmse"]), GRID),
        "extended_grid_meanpx": (min(ext, key=lambda r: r["mean_px"]), EXTENDED_GRID),
        "extended_grid_bmse": (min(ext, key=lambda r: r["bmse"]), EXTENDED_GRID),
    }
    tuning = {"objective_protocol": "overall mean pixel error, per video then over videos (tune_kalman.py)",
              "objective_alternative": "validation balanced MSE (common/loss.balanced_mse), the GRU's selection metric",
              "protocol_grid": GRID, "extended_grid": EXTENDED_GRID, "fixed": FIXED,
              "n_configs": {"protocol": len(proto), "extended": len(ext)}, "seconds": round(t_tune, 1),
              "picks": {}}
    for name, (r, g) in picks.items():
        tuning["picks"][name] = {"params": config_dict(r["cfg"]), "mean_px": r["mean_px"], "bmse": r["bmse"],
                                 **{c: r[c] for c in CATS[1:]}, "edges": edges(r["cfg"], g)}
    for name, rows in (("protocol", proto), ("extended", ext)):
        with open(out / f"kalman_grid_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["accel_px", "meas_px", "partial_px", "min_pixels", "mean_px", "visible", "partial",
                        "hidden", "balanced_mse"])
            for r in sorted(rows, key=lambda r: r["mean_px"]):
                c = r["cfg"]
                w.writerow([c.accel_px, c.meas_px, c.partial_px, c.min_pixels, f"{r['mean_px']:.5f}",
                            f"{r['visible']:.5f}", f"{r['partial']:.5f}", f"{r['hidden']:.5f}", f"{r['bmse']:.4e}"])

    # candidate configs for review (baselines/kalman_config.json is NOT touched)
    for name in ("extended_grid_meanpx", "extended_grid_bmse", "protocol_grid_meanpx"):
        r = picks[name][0]
        doc = {"_note": "CANDIDATE for review; not installed. Selected on data.generator validation seed 200 only.",
               "params": config_dict(r["cfg"]),
               "selection": {"module": gen.__name__, "fingerprint": fp, "data_seed": VAL_SEED, "n_videos": VAL_N,
                             "objective": tuning["objective_protocol"] if name.endswith("meanpx")
                             else tuning["objective_alternative"],
                             "grid": picks[name][1], "fixed": FIXED, "pick": name}}
        (out / f"kalman_config_candidate_{name}.json").write_text(json.dumps(doc, indent=2) + "\n")

    # ---- Steps 3-4: predictions and metrics
    preds = {}
    for s, net in nets.items():
        preds[f"gru_seed{s}"] = gru_predict(frames, net, device=a.device)
    preds["last_pos"] = lp_predict(frames)
    preds["kalman_tuned"] = track(*det, picks["extended_grid_meanpx"][0]["cfg"])
    preds["kalman_tuned_bmse"] = track(*det, picks["extended_grid_bmse"][0]["cfg"])
    preds["kalman_protocol_grid"] = track(*det, picks["protocol_grid_meanpx"][0]["cfg"])
    # the starter-tuned config that baselines/kalman_config.json held before Phase 12 (kept in <exp>)
    original = exp / "kalman_config_original_starter.json"
    if original.exists():
        preds["kalman_installed_starter_cfg"] = track(*det, load_config(original))
    integrity["installed_kalman_config_is_tuned_pick"] = load_config() == picks["extended_grid_meanpx"][0]["cfg"]

    for s in (0, 1, 2):     # recomputed loss must equal the training summary
        got = bmse(preds[f"gru_seed{s}"], xy, vis)
        integrity["checkpoints"][s]["recomputed_val_bmse"] = got
        assert abs(got - integrity["checkpoints"][s]["best_val_loss"]) < 1e-8 + 1e-3 * got

    res, errs = {}, {}
    for name, pr in preds.items():
        res[name], errs[name] = metrics(pr, xy, vis, durations, spans, motion)

    scalar = ["overall", "visible", "partial", "hidden", "median_px", "rmse_px", "within5", "hidden_within5",
              "max_px", "val_balanced_mse"]
    gru_names = [f"gru_seed{s}" for s in (0, 1, 2)]

    def agg(get):
        v = np.array([get(res[g]) for g in gru_names], float)
        return float(v.mean()), float(v.std(ddof=1))

    gru_agg = {k: agg(lambda r, k=k: r[k]) for k in scalar}
    gru_agg["by_duration"] = {d: {k: agg(lambda r, d=d, k=k: r["by_duration"][d][k])
                                  for k in ("overall", "visible", "partial", "hidden", "within5", "last_hidden_px")}
                              for d in DURATIONS}
    gru_agg["recovery_5px_mean_delay"] = agg(lambda r: r["recovery"]["5px"]["all_occluded"]["mean_delay"])

    # ---- paired bootstrap over videos (seed-averaged GRU per-video error)
    masks = {"overall": np.ones_like(vis, bool), "hidden": vis == 0, "partial": (vis > 0) & (vis < 1),
             "visible": vis == 1}
    pv = {name: {c: per_video(errs[name], m) for c, m in masks.items()} for name in preds}
    pv["gru_mean"] = {c: np.mean([pv[g][c] for g in gru_names], 0) for c in masks}
    paired = {}
    for a_, b_ in (("gru_mean", "kalman_tuned"), ("gru_mean", "last_pos"), ("kalman_tuned", "last_pos"),
                   ("gru_seed0", "gru_seed1"), ("gru_seed0", "gru_seed2"), ("gru_seed1", "gru_seed2")):
        paired[f"{a_} - {b_}"] = {c: boot_ci(pv[a_][c] - pv[b_][c]) for c in masks}
    for d in DURATIONS[1:]:
        m = durations == d
        paired[f"gru_mean - kalman_tuned | d={d}"] = {c: boot_ci((pv["gru_mean"][c] - pv["kalman_tuned"][c])[m])
                                                      for c in ("overall", "hidden")}
    # sampling noise of a single seed's overall mean, for comparison with the seed std
    se = {g: boot_ci(pv[g]["overall"]) for g in gru_names}

    # ---- systematic errors on hidden frames (along-track lead/lag)
    systematic = {}
    for name in ("gru_seed0", "gru_seed1", "gru_seed2", "kalman_tuned", "last_pos"):
        ex, ey, h = along_track_hidden(preds[name], xy, vis)
        row = {}
        for mname in ("constant", "smooth"):
            mm = h & (motion == mname)[:, None]
            row[mname] = {"along_track_mean_px": float(ex[mm].mean()), "cross_track_mean_px": float(ey[mm].mean()),
                          "abs_along_px": float(np.abs(ex[mm]).mean()), "abs_cross_px": float(np.abs(ey[mm]).mean())}
        # hidden error vs speed change across the occlusion (smooth videos)
        e = errs[name]
        dv, he = [], []
        for i, sp in enumerate(spans):
            if sp is None:
                continue
            v_in = steps[i, max(sp[0] - 2, 0)]
            v_out = steps[i, min(sp[1], steps.shape[1] - 1)]
            dv.append(v_out - v_in)
            he.append(e[i, sp[0]:sp[1] + 1].mean())
        dv, he = np.array(dv), np.array(he)
        row["corr_hidden_err_vs_abs_speed_change"] = float(np.corrcoef(np.abs(dv), he)[0, 1])
        q = np.quantile(np.abs(dv), [0, 0.5, 0.9, 1])
        row["hidden_err_by_abs_speed_change"] = {
            f"|dv|<={q[1]:.3f}": float(he[np.abs(dv) <= q[1]].mean()),
            f"{q[1]:.3f}<|dv|<={q[2]:.3f}": float(he[(np.abs(dv) > q[1]) & (np.abs(dv) <= q[2])].mean()),
            f"|dv|>{q[2]:.3f}": float(he[np.abs(dv) > q[2]].mean())}
        systematic[name] = row

    # ---- seed-0 diagnosis from training histories
    hist = {s: integrity["checkpoints"][s]["history"] for s in (0, 1, 2)}
    seed_diag = {}
    for s, h in hist.items():
        vl = [e["val_loss"] for e in h]
        tl = [e["train_loss"] for e in h]
        seed_diag[s] = {"val_at_epoch8": vl[7], "best_val_up_to_epoch8": min(vl[:8]),
                        "best_val_up_to_epoch13": min(vl[:13]), "train_at_epoch13": tl[12],
                        "epoch_to_epoch_val_cv_after_5": float(np.std(np.diff(np.log(vl[4:])))),
                        "stopped_at": len(h)}

    # ---- write outputs
    def dump(o):
        if isinstance(o, dict):
            return {str(k): dump(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [dump(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else (str(o) if np.isinf(o) else float(o))
        if isinstance(o, np.integer):
            return int(o)
        return o

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    except OSError:
        git, dirty = "unknown", ""
    prov = {"analysis_script": "baselines/validate_final.py", "analysis_commit": git,
            "working_tree_changes": dirty.splitlines(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "device": a.device,
            "val_split": {"module": gen.__name__, "n": n, "seed": seed, "durations": list(durs),
                          "videos_per_duration": {int(d): int((durations == d).sum()) for d in DURATIONS},
                          "videos_per_motion": {m: int((motion == m).sum()) for m in ("constant", "smooth")}},
            "seed_300": "never generated or read", "runtime_s": None}
    integrity_out = {k: v for k, v in integrity.items()}
    for s in integrity_out["checkpoints"]:
        integrity_out["checkpoints"][s] = {k: v for k, v in integrity_out["checkpoints"][s].items() if k != "history"}
    prov["runtime_s"] = round(time.time() - t_start, 1)
    full = {"provenance": prov, "integrity": integrity_out, "kalman_tuning": tuning, "per_model": res,
            "gru_mean_std": gru_agg, "paired_bootstrap_95ci": paired, "single_seed_overall_ci": se,
            "systematic_hidden_errors": systematic, "seed_diagnosis": seed_diag,
            "training_histories": hist}
    (out / "results.json").write_text(json.dumps(dump(full), indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps(dump({"provenance": prov, "integrity": integrity_out}),
                                                    indent=2) + "\n")

    with open(out / "per_model_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + scalar + ["recovery5_mean_delay", "recovery5_never_frac"])
        for name, r in res.items():
            rec = r["recovery"]["5px"]["all_occluded"]
            w.writerow([name] + [f"{r[k]:.6g}" for k in scalar] + [f"{rec['mean_delay']:.4g}", f"{rec['never_frac']:.4g}"])
        for k, lab in ((0, "gru_mean"), (1, "gru_std_ddof1")):
            w.writerow([lab] + [f"{gru_agg[c][k]:.6g}" for c in scalar] + [f"{gru_agg['recovery_5px_mean_delay'][k]:.4g}", ""])
    with open(out / "per_duration_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "duration", "videos", "overall", "visible", "partial", "hidden", "last_hidden_px",
                    "within5", "recovery5_mean_delay", "recovery5_never_frac", "recovery1_mean_delay",
                    "recovery1_never_frac"])
        for name, r in res.items():
            for d in DURATIONS:
                b = r["by_duration"][d]
                r5 = r["recovery"]["5px"].get(d, {})
                r1 = r["recovery"]["1px"].get(d, {})
                w.writerow([name, d, b["videos"]] + [f"{b[k]:.5g}" for k in
                           ("overall", "visible", "partial", "hidden", "last_hidden_px", "within5")]
                           + [f"{r5.get('mean_delay', float('nan')):.4g}", f"{r5.get('never_frac', float('nan')):.4g}",
                              f"{r1.get('mean_delay', float('nan')):.4g}", f"{r1.get('never_frac', float('nan')):.4g}"])

    charts(out, res, gru_agg, hist, gru_names)
    print(f"wrote {out} in {time.time() - t_start:.1f}s")


def charts(out, res, gru_agg, hist, gru_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = {"last_pos": ("last position", "#8c8c8c"), "kalman_tuned": ("Kalman (tuned)", "#d9822b")}
    gcol = "#2b6cb0"

    # 1. hidden / overall error vs duration
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((axes[0], "hidden", "Fully hidden frames"), (axes[1], "overall", "All frames")):
        ds = DURATIONS[1:] if key == "hidden" else DURATIONS
        for name, (lab, col) in base.items():
            ax.plot(ds, [res[name]["by_duration"][d][key] for d in ds], "o-", color=col, label=lab)
        m = [gru_agg["by_duration"][d][key][0] for d in ds]
        sd = [gru_agg["by_duration"][d][key][1] for d in ds]
        ax.errorbar(ds, m, yerr=sd, fmt="o-", color=gcol, capsize=3, label="CNN-GRU (mean ± sd, 3 seeds)")
        ax.set_xlabel("hidden duration (frames)")
        ax.set_ylabel("mean error (px)")
        ax.set_title(title)
        ax.set_xticks(ds)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation (generator seed 200, 1,000 videos)")
    fig.tight_layout()
    fig.savefig(out / "error_by_duration.png", dpi=130)
    plt.close(fig)

    # 2. event-aligned error around a duration-8 occlusion
    fig, ax = plt.subplots(figsize=(8, 4))
    d = 8
    x = list(range(-4, 0)) + list(range(1, d + 1)) + [d + o for o in range(1, 9)]
    for name, (lab, col) in base.items():
        a = res[name]["aligned"][d]
        ax.plot(x, a["pre_-4..-1"] + a["hidden_1..d"] + a["post_+1..+8"], "o-", ms=3, color=col, label=lab)
    for g, ls in zip(gru_names, ("-", "--", ":")):
        a = res[g]["aligned"][d]
        ax.plot(x, a["pre_-4..-1"] + a["hidden_1..d"] + a["post_+1..+8"], ls, color=gcol, label=g)
    ax.axvspan(0.5, d + 0.5, color="k", alpha=0.07, label="fully hidden")
    ax.set_xlabel("frame relative to occlusion (1..8 = fully hidden frames)")
    ax.set_ylabel("mean error (px)")
    ax.set_title("Error around an 8-frame occlusion (validation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "error_aligned_d8.png", dpi=130)
    plt.close(fig)

    # 3. training curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for (s, h), col in zip(hist.items(), ("#2b6cb0", "#38a169", "#805ad5")):
        ep = [e["epoch"] for e in h]
        ax.semilogy(ep, [e["val_loss"] for e in h], "o-", ms=3, color=col, label=f"seed {s} val")
        ax.semilogy(ep, [e["train_loss"] for e in h], "--", color=col, alpha=0.6, label=f"seed {s} train")
    ax.set_xlabel("epoch")
    ax.set_ylabel("balanced MSE (normalized units)")
    ax.set_title("CNN-GRU training curves (early stop, patience 5)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)

    # 4. category comparison bars
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = ["overall", "visible", "partial", "hidden"]
    w = 0.27
    xs = np.arange(len(cats))
    for i, (name, (lab, col)) in enumerate(base.items()):
        ax.bar(xs + (i - 1) * w, [res[name][c] for c in cats], w, color=col, label=lab)
    ax.bar(xs + w, [gru_agg[c][0] for c in cats], w, yerr=[gru_agg[c][1] for c in cats], color=gcol,
           capsize=3, label="CNN-GRU (mean ± sd)")
    ax.set_xticks(xs, cats)
    ax.set_ylabel("mean error (px)")
    ax.set_title("Validation error by visibility")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "error_by_visibility.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
