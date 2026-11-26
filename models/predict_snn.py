"""Write official test-bank predictions for the SNN and its memory ablation.

Outputs (CONTRACT.md section 3):
    preds/snn_seed<k>.npz          normal SNN (states carried across all 40 frames)
    preds/snn_reset_seed<k>.npz    onset memory ablation (states reset once at first
                                   fully hidden frame, then rollout continues)
    preds/snn_reset_every_seed<k>.npz  optional every-frame reset (--reset-mode every_frame/both)

Both run on data.generator.iter_split("test") in dataset order, pass only `frames`
to the model, and save data_tag = OFFICIAL_TEST_TAG with spikes and n_params.

Examples:
    python -m models.predict_snn --seed 0 --device mps --verify
    python -m models.predict_snn --seed 0 --checkpoint checkpoints/smoke/snn_seed0.pt --out-dir /tmp/snn_smoke --device cpu --verify
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import data.generator as generator
from baselines import data_cache
from common.io import load_preds, save_preds
from data.generator import OFFICIAL_TEST_TAG, iter_split, iter_test_predictions
from models.snn import EXPECTED_PARAMS, count_params
from models.train_snn import load_checkpoint, resolve_device

RESET_MODES = ("onset", "every_frame", "both")


@torch.no_grad()
def predict_split(model, split_iter, device, reset_mode="none"):
    """Run inference over an iterator of split dicts using `frames` only.

    reset_mode:
        "none"        normal rollout (states carried across all frames)
        "onset"       reset all states once at the first frame with visible_frac <= 0
                      (visible_frac is inspected here only, never passed to the model)
        "every_frame" reset all states before every frame
    """
    if reset_mode not in ("none", "onset", "every_frame"):
        raise ValueError(f"unknown reset_mode: {reset_mode}")

    was_training = model.training
    model.eval()
    preds_all, spikes_all = [], []

    for batch in split_iter:
        frames = torch.from_numpy(np.ascontiguousarray(batch["frames"], np.float32)).to(device)
        if reset_mode == "none":
            pred_xy, spikes = model(frames)
        elif reset_mode == "every_frame":
            pred_xy, spikes = model(frames, reset_each_frame=True)
        else:
            B, T = frames.shape[:2]
            vis = np.asarray(batch["visible_frac"])
            state = model.init_state(B, device)
            already_reset = np.zeros(B, dtype=bool)
            preds, spks = [], []

            for t in range(T):
                reset_mask = (vis[:, t] <= 0) & (~already_reset)
                if reset_mask.any():
                    already_reset |= reset_mask
                    keep = torch.from_numpy(~reset_mask).to(device=device)
                    fresh = model.init_state(B, device)
                    state = tuple(
                        torch.where(keep.view(B, *([1] * (s.ndim - 1))), s, f)
                        for s, f in zip(state, fresh)
                    )
                xy_t, state, spk_t = model.step(frames[:, t], state)
                preds.append(xy_t)
                spks.append(spk_t)

            pred_xy = torch.stack(preds, dim=1)
            spikes = torch.stack(spks, dim=1)

        preds_all.append(pred_xy.cpu().numpy())
        spikes_all.append(spikes.cpu().numpy())

    model.train(was_training)
    return (
        np.concatenate(preds_all, axis=0).astype(np.float32),
        np.concatenate(spikes_all, axis=0).astype(np.float32),
    )


def _check_provenance(payload, checkpoint_path, allow_data_mismatch=False):
    cfg = payload.get("config", {})
    trained_module = cfg.get("data_module")
    trained_fp = cfg.get("data_fingerprint")
    current_fp = data_cache.source_fingerprint(generator)
    if not allow_data_mismatch:
        if trained_module is not None and trained_module != generator.__name__:
            raise ValueError(
                f"{checkpoint_path} was trained on {trained_module}, not {generator.__name__}. "
                "Retrain on generator or pass --allow-data-mismatch."
            )
        if trained_fp is not None and trained_fp != current_fp:
            raise ValueError(
                f"{generator.__name__} changed since {checkpoint_path} was trained. "
                "Retrain or pass --allow-data-mismatch."
            )


def write_predictions(
    seed=0,
    checkpoint="",
    out_dir="preds",
    batch_size=64,
    device="auto",
    reset_mode="onset",
    allow_data_mismatch=False,
):
    """Load checkpoint for `seed` and write snn + reset ablation .npz files."""
    if reset_mode not in RESET_MODES:
        raise ValueError(f"reset_mode must be one of {RESET_MODES}, got {reset_mode}")

    dev = resolve_device(device)
    ckpt_path = checkpoint or f"checkpoints/snn_seed{seed}.pt"
    model, payload = load_checkpoint(ckpt_path, dev)

    ckpt_seed = payload.get("config", {}).get("seed")
    if ckpt_seed is not None and ckpt_seed != seed:
        raise ValueError(f"checkpoint {ckpt_path} has seed {ckpt_seed}, expected --seed {seed}")
    _check_provenance(payload, ckpt_path, allow_data_mismatch=allow_data_mismatch)

    n_params = count_params(model)
    assert n_params == EXPECTED_PARAMS, n_params

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Normal SNN: states start at 0 per video and carry across all 40 frames.
    pred_xy, spikes = predict_split(
        model, iter_split("test", batch_size=batch_size), dev, reset_mode="none"
    )
    snn_path = out_dir / f"snn_seed{seed}.npz"
    save_preds(
        snn_path,
        pred_xy,
        model="snn",
        seed=seed,
        data_tag=OFFICIAL_TEST_TAG,
        spikes=spikes,
        n_params=n_params,
    )
    written.append(snn_path)

    # Onset ablation: states reset once at the first fully hidden frame (visible_frac == 0).
    if reset_mode in ("onset", "both"):
        pred_onset, spk_onset = predict_split(
            model, iter_split("test", batch_size=batch_size), dev, reset_mode="onset"
        )
        onset_path = out_dir / f"snn_reset_seed{seed}.npz"
        save_preds(
            onset_path,
            pred_onset,
            model="snn_reset",
            seed=seed,
            data_tag=OFFICIAL_TEST_TAG,
            spikes=spk_onset,
            n_params=n_params,
        )
        written.append(onset_path)

    # Optional every-frame ablation: states reset before every frame.
    if reset_mode in ("every_frame", "both"):
        pred_every, spk_every = predict_split(
            model, iter_split("test", batch_size=batch_size), dev, reset_mode="every_frame"
        )
        every_path = out_dir / f"snn_reset_every_seed{seed}.npz"
        save_preds(
            every_path,
            pred_every,
            model="snn_reset_every",
            seed=seed,
            data_tag=OFFICIAL_TEST_TAG,
            spikes=spk_every,
            n_params=n_params,
        )
        written.append(every_path)

    return tuple(written)


def verify_predictions(path):
    """Verify a saved prediction file using data.generator.iter_test_predictions."""
    saved = load_preds(path)
    assert saved["data_tag"] == OFFICIAL_TEST_TAG
    assert int(saved["n_params"]) == EXPECTED_PARAMS
    assert saved["spikes"].shape == saved["pred_xy"].shape[:2]

    vis_errs, hid_errs, hid_spks = [], [], []
    offset = 0
    for pred_xy, true_xy, visible_frac in iter_test_predictions(path):
        n = len(pred_xy)
        batch_spks = saved["spikes"][offset:offset + n]
        offset += n

        err_px = np.sqrt(((pred_xy - true_xy) ** 2).sum(axis=-1)) * 64.0
        vis_mask = visible_frac > 0
        hid_mask = visible_frac <= 0
        if vis_mask.any():
            vis_errs.append(err_px[vis_mask])
        if hid_mask.any():
            hid_errs.append(err_px[hid_mask])
            hid_spks.append(batch_spks[hid_mask])

    vis_mean = float(np.concatenate(vis_errs).mean())
    hid_mean = float(np.concatenate(hid_errs).mean())
    spk_mean = float(saved["spikes"].mean())
    hid_spk_mean = float(np.concatenate(hid_spks).mean())
    print(
        f"verified {path}: rows={offset}, model={saved['model']}, seed={saved['seed']}, "
        f"tag={saved['data_tag']}, n_params={int(saved['n_params']):,}, "
        f"spikes_shape={saved['spikes'].shape}, mean_spk/frame={spk_mean:.1f}, "
        f"hid_spk/frame={hid_spk_mean:.1f}, vis_err={vis_mean:.2f}px, hid_err={hid_mean:.2f}px"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="training seed (0, 1, or 2)")
    p.add_argument("--checkpoint", default="", help="default: checkpoints/snn_seed<seed>.pt")
    p.add_argument("--out-dir", default="preds")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--reset-mode",
        default="onset",
        choices=RESET_MODES,
        help="ablation mode: onset (default -> snn_reset), every_frame (-> snn_reset_every), or both",
    )
    p.add_argument("--verify", action="store_true", help="verify outputs with iter_test_predictions")
    p.add_argument("--allow-data-mismatch", action="store_true")
    a = p.parse_args(argv)

    written = write_predictions(
        seed=a.seed,
        checkpoint=a.checkpoint,
        out_dir=a.out_dir,
        batch_size=a.batch_size,
        device=a.device,
        reset_mode=a.reset_mode,
        allow_data_mismatch=a.allow_data_mismatch,
    )
    for path in written:
        print(f"wrote {path}")
    if a.verify:
        for path in written:
            verify_predictions(path)


if __name__ == "__main__":
    main()
