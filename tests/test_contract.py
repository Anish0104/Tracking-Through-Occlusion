"""Checks that the shared pieces follow CONTRACT.md. Run: python -m pytest -q"""

import numpy as np
import torch

from common.fake import fake_preds
from common.io import load_preds, save_preds
from common.loss import balanced_mse
from data.starter import make_batch


def test_starter_shapes_and_ranges():
    b = make_batch(n=8, seed=0)
    assert b["frames"].shape == (8, 40, 64, 64) and b["frames"].dtype == np.float32
    assert b["xy"].shape == (8, 40, 2)
    assert b["visible_frac"].shape == (8, 40)
    assert 0 <= b["frames"].min() and b["frames"].max() <= 1
    assert 0 < b["xy"].min() and b["xy"].max() < 1


def test_starter_has_occlusion_and_is_reproducible():
    b = make_batch(n=32, seed=1)
    assert (b["visible_frac"] == 0).any(), "some frames should be fully hidden"
    assert np.array_equal(b["xy"], make_batch(n=32, seed=1)["xy"])


def test_loss_is_zero_for_perfect_predictions():
    b = make_batch(n=4, seed=2)
    xy = torch.tensor(b["xy"])
    assert balanced_mse(xy, xy, torch.tensor(b["visible_frac"])).item() == 0.0


def test_preds_roundtrip(tmp_path):
    b = make_batch(n=4, seed=3)
    p = fake_preds(b["xy"], b["visible_frac"])
    f = tmp_path / "fake_seed0.npz"
    save_preds(f, p, model="fake", seed=0, data_tag="starter_test")
    out = load_preds(f)
    assert out["model"] == "fake" and out["seed"] == 0
    assert np.allclose(out["pred_xy"], p)
