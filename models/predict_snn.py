"""Write official test-bank predictions for the SNN and its state-reset ablation.

Outputs (CONTRACT.md section 3):
    preds/snn_seed<k>.npz        normal SNN (membrane states carried across all 40 frames)
    preds/snn_reset_seed<k>.npz  memory ablation (all membrane states reset every frame)

Both files run on data.generator.iter_split("test") in dataset order, pass only
`frames` to the model, and save data_tag = OFFICIAL_TEST_TAG with spikes and n_params.

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


@torch.no_grad()
def predict_split(model, split_iter, device, reset_each_frame=False):
    """Run inference over an iterator of split dicts using `frames` only.

    Returns (pred_xy float32 [N, T, 2], spikes float32 [N, T]).
    """
    was_training = model.training
    model.eval()
    preds_all, spikes_all = [], []
    for batch in split_iter:
        frames = torch.from_numpy(np.ascontiguousarray(batch["frames"], np.float32)).to(device)
        pred_xy, spikes = model(frames, reset_each_frame=reset_each_frame)
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
    allow_data_mismatch=False,
):
    """Load checkpoint for `seed` and write snn + snn_reset .npz files on official test bank."""
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

    # 1) Normal SNN: membrane states start at 0 per video and carry across all 40 frames.
    pred_xy, spikes = predict_split(
        model, iter_split("test", batch_size=batch_size), dev, reset_each_frame=False
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

    # 2) State-reset ablation: same weights, all membrane states reset to 0 at every frame.
    pred_xy_reset, spikes_reset = predict_split(
        model, iter_split("test", batch_size=batch_size), dev, reset_each_frame=True
    )
    reset_path = out_dir / f"snn_reset_seed{seed}.npz"
    save_preds(
        reset_path,
        pred_xy_reset,
        model="snn_reset",
        seed=seed,
        data_tag=OFFICIAL_TEST_TAG,
        spikes=spikes_reset,
        n_params=n_params,
    )

    return snn_path, reset_path


def verify_predictions(path):
    """Verify a saved prediction file using data.generator.iter_test_predictions."""
    saved = load_preds(path)
    assert saved["data_tag"] == OFFICIAL_TEST_TAG
    assert int(saved["n_params"]) == EXPECTED_PARAMS
    assert saved["spikes"].shape == saved["pred_xy"].shape[:2]

    vis_sq_errs, hid_sq_errs = [], []
    total_rows = 0
    for pred_xy, true_xy, visible_frac in iter_test_predictions(path):
        total_rows += len(pred_xy)
        err_px = np.sqrt(((pred_xy - true_xy) ** 2).sum(axis=-1)) * 64.0
        vis_mask = visible_frac > 0
        hid_mask = visible_frac <= 0
        if vis_mask.any():
            vis_sq_errs.append(err_px[vis_mask])
        if hid_mask.any():
            hid_sq_errs.append(err_px[hid_mask])

    vis_mean = float(np.concatenate(vis_sq_errs).mean())
    hid_mean = float(np.concatenate(hid_sq_errs).mean())
    spk_mean = float(saved["spikes"].mean())
    print(
        f"verified {path}: rows={total_rows}, model={saved['model']}, seed={saved['seed']}, "
        f"tag={saved['data_tag']}, n_params={int(saved['n_params']):,}, "
        f"spikes_shape={saved['spikes'].shape}, mean_spk/frame={spk_mean:.1f}, "
        f"vis_err={vis_mean:.2f}px, hid_err={hid_mean:.2f}px"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="training seed (0, 1, or 2)")
    p.add_argument("--checkpoint", default="", help="default: checkpoints/snn_seed<seed>.pt")
    p.add_argument("--out-dir", default="preds")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--verify", action="store_true", help="verify outputs with iter_test_predictions")
    p.add_argument("--allow-data-mismatch", action="store_true")
    a = p.parse_args(argv)

    snn_path, reset_path = write_predictions(
        seed=a.seed,
        checkpoint=a.checkpoint,
        out_dir=a.out_dir,
        batch_size=a.batch_size,
        device=a.device,
        allow_data_mismatch=a.allow_data_mismatch,
    )
    print(f"wrote {snn_path}")
    print(f"wrote {reset_path}")
    if a.verify:
        verify_predictions(snn_path)
        verify_predictions(reset_path)


if __name__ == "__main__":
    main()
