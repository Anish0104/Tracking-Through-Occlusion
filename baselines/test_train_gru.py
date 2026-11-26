"""Tests for CNN-GRU training, checkpointing and inference (tiny data, CPU, seconds).
Run: python -m pytest -q"""

import numpy as np
import pytest
import torch

from baselines import predict as predict_mod
from baselines import train_gru
from baselines.gru import EXPECTED_PARAMS, build_model, predict
from baselines.train_gru import (EarlyStopping, TrainConfig, evaluate, iterate_batches,
                                 load_checkpoint, load_split, save_checkpoint, train,
                                 train_one_epoch)
from common.io import load_preds
from common.loss import balanced_mse
from data.starter import make_batch


@pytest.fixture(scope="module")
def tiny():
    return {"train": make_batch(n=8, seed=100), "val": make_batch(n=4, seed=200)}


def _cfg(tmp_path, **kw):
    base = dict(seed=0, n_train=8, n_val=4, epochs=1, batch_size=4, device="cpu",
                checkpoint=str(tmp_path / "gru_seed0.pt"), log_every=0)
    base.update(kw)
    return TrainConfig(**base)


def _torch_batch(b):
    return [torch.from_numpy(b[k]) for k in ("frames", "xy", "visible_frac")]


# ---------------------------------------------------------------- contract values

def test_defaults_match_contract():
    cfg = TrainConfig()
    assert (cfg.lr, cfg.batch_size, cfg.epochs, cfg.patience) == (1e-3, 16, 30, 5)
    assert (cfg.train_seed, cfg.n_train, cfg.val_seed, cfg.n_val) == (100, 5000, 200, 1000)
    assert cfg.checkpoint == "checkpoints/gru_seed0.pt"
    assert TrainConfig(seed=2).checkpoint == "checkpoints/gru_seed2.pt"


def test_training_uses_balanced_mse_and_adam():
    import inspect
    src = inspect.getsource(train_gru)
    assert "from common.loss import balanced_mse" in src
    assert "torch.optim.Adam(model.parameters(), lr=cfg.lr)" in src


# ---------------------------------------------------------------- split usage

def test_train_and_validation_splits_use_contract_seeds(tmp_path, monkeypatch):
    calls = []

    def fake_make_batch(n, seed):
        calls.append((n, seed))
        return make_batch(n=n, seed=seed)

    monkeypatch.setattr(train_gru, "get_make_batch", lambda source: fake_make_batch)
    train(_cfg(tmp_path))
    assert calls == [(8, 100), (4, 200)]          # train seed 100, validation seed 200, never 300


@pytest.mark.parametrize("kw", [dict(train_seed=300), dict(val_seed=300),
                                dict(train_seed=100, val_seed=100)])
def test_rejects_test_bank_and_overlapping_seeds(kw):
    with pytest.raises(ValueError):
        TrainConfig(**kw)


def test_data_source_is_pluggable():
    assert train_gru.get_make_batch("starter") is make_batch
    assert train_gru.get_make_batch("data.starter") is make_batch
    assert train_gru.DATA_SOURCES["generator"] == "data.generator"   # Krisha's final generator


def test_split_cache_matches_generation_and_is_memmapped(tmp_path):
    fresh = load_split("starter", 3, 200)
    cached = load_split("starter", 3, 200, cache_dir=tmp_path)    # writes
    again = load_split("starter", 3, 200, cache_dir=tmp_path)     # reads
    for k in fresh:
        backing = again[k].codes if k == "frames" else again[k]     # frames: uint8 palette codes
        assert isinstance(backing, np.memmap)
        np.testing.assert_array_equal(fresh[k], cached[k])
        np.testing.assert_array_equal(fresh[k], again[k])


def test_batches_cover_split_exactly_once(tiny):
    order = np.random.default_rng(0).permutation(8)
    seen = [f for f, _, _ in iterate_batches(tiny["train"], 3, order)]
    assert [len(f) for f in seen] == [3, 3, 2]
    got = torch.cat(seen).numpy()
    np.testing.assert_array_equal(got, tiny["train"]["frames"][order])


# ---------------------------------------------------------------- model sees frames only

def test_model_receives_only_frames(tiny):
    model = build_model(seed=0)
    seen = []
    model.register_forward_pre_hook(lambda m, args, kwargs: seen.append((args, kwargs)),
                                    with_kwargs=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_one_epoch(model, tiny["train"], opt, 4, torch.device("cpu"), np.random.default_rng(0))
    evaluate(model, tiny["val"], 4, torch.device("cpu"))
    predict(tiny["val"]["frames"], model)
    assert len(seen) == 2 + 1 + 1
    for args, kwargs in seen:
        assert len(args) == 1 and not kwargs
        assert args[0].ndim == 4 and tuple(args[0].shape[-2:]) == (64, 64)
        # pixel values only: the disk/barrier/background levels of CONTRACT.md section 1
        assert set(np.unique(args[0].numpy())) <= {0.0, np.float32(0.4), 1.0}


# ---------------------------------------------------------------- loss, gradients, learning

def test_loss_and_gradients_are_finite(tiny):
    model = build_model(seed=0)
    frames, xy, vis = _torch_batch(tiny["train"])
    loss = balanced_mse(model(frames)[0], xy, vis)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert sum(p.grad.abs().sum().item() for p in model.parameters()) > 0


def test_single_batch_training_reduces_loss(tiny):
    torch.manual_seed(0)
    model = build_model(seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    frames, xy, vis = _torch_batch(tiny["train"])
    losses = []
    for _ in range(25):
        loss = balanced_mse(model(frames)[0], xy, vis)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[1] < losses[0]                  # one step already helps
    assert losses[-1] < 0.5 * losses[0]           # and it overfits a single batch


def test_validation_loss_independent_of_batch_size(tiny):
    model = build_model(seed=0)
    a = evaluate(model, tiny["val"], 1, torch.device("cpu"))
    b = evaluate(model, tiny["val"], 3, torch.device("cpu"))
    assert a == pytest.approx(b, rel=1e-5)


# ---------------------------------------------------------------- early stopping

def test_early_stopping_counts_epochs_without_improvement():
    es = EarlyStopping(patience=2)
    assert es.update(1.0, 1) and not es.should_stop
    assert not es.update(1.1, 2) and not es.should_stop
    assert es.update(0.9, 3) and es.bad_epochs == 0      # improvement resets the counter
    assert not es.update(0.95, 4)
    assert not es.update(0.9, 5) and es.should_stop      # equal is not an improvement
    assert (es.best, es.best_epoch) == (0.9, 3)


def test_early_stopping_min_delta_and_disabled():
    es = EarlyStopping(patience=1, min_delta=0.1)
    es.update(1.0, 1)
    assert not es.update(0.95, 2) and es.should_stop
    off = EarlyStopping(patience=None)
    for e in range(10):
        off.update(1.0, e)
    assert not off.should_stop


def test_training_stops_early_and_keeps_best_checkpoint(tmp_path, tiny, monkeypatch):
    val_losses = iter([0.5, 0.3, 0.4, 0.45, 0.6, 0.7])
    monkeypatch.setattr(train_gru, "evaluate", lambda *a, **k: next(val_losses))
    out = train(_cfg(tmp_path, epochs=6, patience=2), tiny["train"], tiny["val"])
    assert out["stopped_early"] and out["epochs_run"] == 4
    assert out["best_epoch"] == 2 and out["best_val_loss"] == 0.3
    _, payload = load_checkpoint(tmp_path / "gru_seed0.pt")
    assert payload["epoch"] == 2 and payload["val_loss"] == 0.3
    assert (tmp_path / "gru_seed0.json").exists()


# ---------------------------------------------------------------- checkpoints

def test_checkpoint_round_trip(tmp_path, tiny):
    model = build_model(seed=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_one_epoch(model, tiny["train"], opt, 4, torch.device("cpu"), np.random.default_rng(1))
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=3, val_loss=0.25, history=[{"epoch": 3}],
                    config={"seed": 1})
    loaded, payload = load_checkpoint(path)
    assert not loaded.training
    assert payload["n_params"] == EXPECTED_PARAMS == 1_067_064
    assert (payload["epoch"], payload["val_loss"], payload["config"]["seed"]) == (3, 0.25, 1)
    for (k, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), k
    frames = tiny["val"]["frames"]
    np.testing.assert_array_equal(predict(frames, model), predict(frames, loaded))
    opt2 = torch.optim.Adam(loaded.parameters(), lr=1e-3)
    opt2.load_state_dict(payload["optimizer_state"])            # resumable


def test_load_checkpoint_rejects_non_gru(tmp_path):
    torch.save({"model": "snn"}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "x.pt")


# ---------------------------------------------------------------- reproducibility

def test_seed_initialization_is_reproducible():
    a, b, c = build_model(seed=0), build_model(seed=0), build_model(seed=1)
    for (k, x), (_, y), (_, z) in zip(a.state_dict().items(), b.state_dict().items(),
                                      c.state_dict().items()):
        assert torch.equal(x, y), k
    assert not all(torch.equal(x, z) for x, z in zip(a.state_dict().values(),
                                                    c.state_dict().values()))


def test_training_run_is_reproducible(tmp_path, tiny):
    r1 = train(_cfg(tmp_path / "a", epochs=2), tiny["train"], tiny["val"])
    r2 = train(_cfg(tmp_path / "b", epochs=2), tiny["train"], tiny["val"])
    strip = lambda h: [(e["train_loss"], e["val_loss"]) for e in h]
    assert strip(r1["history"]) == strip(r2["history"])
    m1, _ = load_checkpoint(tmp_path / "a" / "gru_seed0.pt")
    m2, _ = load_checkpoint(tmp_path / "b" / "gru_seed0.pt")
    for x, y in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(x, y)


# ---------------------------------------------------------------- prediction files

def test_gru_prediction_file_format(tmp_path, tiny):
    ckpt = tmp_path / "gru_seed0.pt"
    train(_cfg(tmp_path, checkpoint=str(ckpt)), tiny["train"], tiny["val"])
    out = predict_mod.run(model="gru", n=5, data_seed=200, seed=0, out_dir=tmp_path / "preds",
                          checkpoint=str(ckpt), device="cpu")
    assert out.name == "gru_seed0.npz"
    p = load_preds(out)
    assert p["pred_xy"].shape == (5, 40, 2) and p["pred_xy"].dtype == np.float32
    assert np.isfinite(p["pred_xy"]).all()
    assert ((p["pred_xy"] > 0) & (p["pred_xy"] < 1)).all()       # sigmoid readout
    assert (p["model"], p["seed"], p["n_params"]) == ("gru", 0, 1_067_064)
    assert p["data_tag"] == "starter_test_s200"
    model, _ = load_checkpoint(ckpt)
    np.testing.assert_allclose(p["pred_xy"], predict(make_batch(5, 200)["frames"], model))


def test_gru_prediction_rejects_wrong_seed(tmp_path, tiny):
    ckpt = tmp_path / "gru_seed0.pt"
    train(_cfg(tmp_path, checkpoint=str(ckpt)), tiny["train"], tiny["val"])
    with pytest.raises(ValueError):
        predict_mod.run(model="gru", n=2, data_seed=200, seed=1, out_dir=tmp_path,
                        checkpoint=str(ckpt))


def test_dev_metrics_cli_scores_baselines_and_gru(tmp_path, tiny, capsys):
    from baselines import dev_metrics
    ckpt = tmp_path / "gru_seed0.pt"
    train(_cfg(tmp_path, checkpoint=str(ckpt)), tiny["train"], tiny["val"])
    capsys.readouterr()                           # drop the training log
    dev_metrics.main(["--n", "4", "--models", "last_pos", "kalman", "gru",
                      "--gru-checkpoint", str(ckpt), "--allow-data-mismatch"])   # generator-tuned Kalman
    out = capsys.readouterr().out
    for m in ("last_pos", "kalman", "gru"):
        assert sum(line.startswith(m + " ") for line in out.splitlines()) == 4   # one row per category
    with pytest.raises(SystemExit):
        dev_metrics.main(["--n", "2", "--data-seed", "300"])
