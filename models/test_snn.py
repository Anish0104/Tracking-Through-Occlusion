"""Pytest suite for the recurrent SNN, training loop, and prediction files."""

import inspect

import numpy as np
import pytest
import torch

from common.io import load_preds, save_preds
from common.loss import balanced_mse
from data.generator import OFFICIAL_TEST_TAG, iter_test_predictions, make_batch
from models.lif import LIF
from models.predict_snn import predict_split, write_predictions
from models.snn import EXPECTED_PARAMS, RecurrentSNN, build_model, count_params, param_breakdown
from models.train_snn import TrainConfig, evaluate, train, train_one_epoch


@pytest.fixture(scope="module")
def model():
    return build_model(seed=0).eval()


@pytest.fixture(scope="module")
def sample_batch():
    b = make_batch(n=4, seed=1, durations=(0, 4))
    return {k: torch.from_numpy(v) for k, v in b.items()}


def test_lif_spike_and_surrogate_grad():
    lif = LIF(beta=0.9, threshold=1.0, slope=25.0)
    assert count_params(lif) == 0

    current = torch.tensor([0.25], requires_grad=True)
    mem = torch.zeros(1)
    spks = []
    for _ in range(10):
        spk, mem = lif(current, mem)
        spks.append(spk)
    total = torch.stack(spks).sum()
    assert total.item() == 2.0
    total.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all() and current.grad.item() > 0


def test_exact_parameter_count(model):
    assert count_params(model) == EXPECTED_PARAMS == 1_070_274
    assert param_breakdown(model) == {
        "conv1": 160,
        "lif1": 0,
        "conv2": 4_640,
        "lif2": 0,
        "fc_in": 1_048_704,
        "fc_rec": 16_512,
        "lif3": 0,
        "readout": 258,
    }


def test_output_shape_range_and_dtype(model, sample_batch):
    frames_4d = sample_batch["frames"]
    frames_5d = frames_4d.unsqueeze(2)

    pred_4d, spk_4d = model(frames_4d)
    pred_5d, spk_5d = model(frames_5d)

    assert torch.allclose(pred_4d, pred_5d)
    assert torch.allclose(spk_4d, spk_5d)

    assert pred_5d.shape == (4, 40, 2) and pred_5d.dtype == torch.float32
    assert spk_5d.shape == (4, 40) and spk_5d.dtype == torch.float32
    assert torch.isfinite(pred_5d).all()
    assert (pred_5d > 0).all() and (pred_5d < 1).all()
    assert (spk_5d >= 0).all()

    with pytest.raises(ValueError):
        model(torch.zeros(2, 64, 64))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 10, 2, 64, 64))


def test_causal_future_frames_do_not_affect_past(model, sample_batch):
    frames = sample_batch["frames"][:2]
    t_cut = 15
    altered = frames.clone()
    altered[:, t_cut + 1:] = 0.0

    pred_a, spk_a = model(frames)
    pred_b, spk_b = model(altered)
    assert torch.allclose(pred_a[:, :t_cut + 1], pred_b[:, :t_cut + 1], atol=1e-6)
    assert torch.allclose(spk_a[:, :t_cut + 1], spk_b[:, :t_cut + 1], atol=1e-6)


@torch.no_grad()
def test_state_resets_between_videos(model, sample_batch):
    frames = sample_batch["frames"]
    for t in model.init_state(3):
        assert torch.equal(t, torch.zeros_like(t))

    pred_fresh, spk_fresh = model(frames[1:2])
    # Run another video and obtain its final state via step()
    state = None
    for t in range(frames.shape[1]):
        _, state, _ = model.step(frames[0:1, t], state)

    pred_leaked, _ = model(frames[1:2], state=state)
    assert not torch.allclose(pred_leaked, pred_fresh)

    # Without passing `state`, calling the model on video 1 again still resets to zeros.
    pred_repeat, spk_repeat = model(frames[1:2])
    assert torch.equal(pred_repeat, pred_fresh)
    assert torch.equal(spk_repeat, spk_fresh)


@torch.no_grad()
def test_onset_reset_at_first_hidden_frame_only(model):
    # Mix a zero-occlusion video (index 0) and occluded videos (duration 4).
    vis_batch = make_batch(1, seed=5, durations=(0,))
    occ_batch = make_batch(3, seed=6, durations=(4,))
    mixed = {
        k: np.concatenate([vis_batch[k], occ_batch[k]], axis=0)
        for k in ("frames", "xy", "visible_frac")
    }

    pred_snn, spk_snn = predict_split(model, [mixed], torch.device("cpu"), reset_mode="none")
    pred_onset, spk_onset = predict_split(model, [mixed], torch.device("cpu"), reset_mode="onset")

    # Video 0 has no hidden frames: never reset, must match snn identically.
    assert (mixed["visible_frac"][0] > 0).all()
    assert np.array_equal(pred_onset[0], pred_snn[0])
    assert np.array_equal(spk_onset[0], spk_snn[0])

    # Videos 1..3 have hidden frames: match snn strictly before first hidden frame,
    # reset once at first_hid, then continue normally afterwards.
    frames_t = torch.from_numpy(mixed["frames"])
    for idx in range(1, 4):
        first_hid = int(np.flatnonzero(mixed["visible_frac"][idx] <= 0)[0])
        assert first_hid > 0
        assert np.array_equal(pred_onset[idx, :first_hid], pred_snn[idx, :first_hid])
        assert np.array_equal(spk_onset[idx, :first_hid], spk_snn[idx, :first_hid])

        # Verify against manual single-video rollout resetting only at first_hid.
        state = None
        ref_preds, ref_spks = [], []
        for t in range(frames_t.shape[1]):
            if t == first_hid:
                state = None
            xy_t, state, spk_t = model.step(frames_t[idx:idx + 1, t], state)
            ref_preds.append(xy_t[0].numpy())
            ref_spks.append(float(spk_t[0].item()))
        assert np.allclose(pred_onset[idx], np.stack(ref_preds), atol=1e-6)
        assert np.allclose(spk_onset[idx], np.asarray(ref_spks, dtype=np.float32), atol=1e-6)
        assert not np.allclose(pred_onset[idx, first_hid:], pred_snn[idx, first_hid:])


@torch.no_grad()
def test_reset_each_frame_ablation_matches_memoryless_steps(model, sample_batch):
    frames = sample_batch["frames"][:2, :10]
    pred_reset, spk_reset = model(frames, reset_each_frame=True)
    step_preds, step_spks = [], []
    for t in range(frames.shape[1]):
        xy_t, _, spk_t = model.step(frames[:, t], state=None)
        step_preds.append(xy_t)
        step_spks.append(spk_t)
    assert torch.allclose(pred_reset, torch.stack(step_preds, dim=1), atol=1e-6)
    assert torch.allclose(spk_reset, torch.stack(step_spks, dim=1), atol=1e-6)


def test_model_never_receives_labels(sample_batch):
    for fn in (RecurrentSNN.forward, RecurrentSNN.step, predict_split):
        params = list(inspect.signature(fn).parameters)
        assert "xy" not in params and "visible_frac" not in params

    seen_forward, seen_step = [], []

    class FrameOnlyWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def init_state(self, batch_size, device=None):
            return self.inner.init_state(batch_size, device=device)

        def step(self, frame, state=None):
            seen_step.append(frame.detach().cpu())
            return self.inner.step(frame, state=state)

        def forward(self, frames, **kwargs):
            seen_forward.append(frames.detach().cpu())
            return self.inner(frames, **kwargs)

    m = FrameOnlyWrapper(build_model(seed=0))
    raw = {k: v.numpy() for k, v in sample_batch.items()}

    # Inference paths: normal (none) and onset reset (uses step)
    predict_split(m, [raw], torch.device("cpu"), reset_mode="none")
    assert len(seen_forward) == 1
    assert torch.equal(seen_forward[0], sample_batch["frames"])

    predict_split(m, [raw], torch.device("cpu"), reset_mode="onset")
    assert len(seen_step) == sample_batch["frames"].shape[1]
    assert torch.equal(torch.stack(seen_step, dim=1), sample_batch["frames"])

    # Training and validation paths: train_one_epoch and evaluate
    seen_forward.clear()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    train_one_epoch(m, raw, opt, batch_size=4, device=torch.device("cpu"), rng=np.random.default_rng(0))
    evaluate(m, raw, batch_size=4, device=torch.device("cpu"))
    assert len(seen_forward) == 2
    assert all(arg.shape == sample_batch["frames"].shape for arg in seen_forward)


def test_surrogate_backprop_reaches_every_parameter(sample_batch):
    m = build_model(seed=0).train()
    pred, _ = m(sample_batch["frames"])
    loss = balanced_mse(pred, sample_batch["xy"], sample_batch["visible_frac"])
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None, f"missing grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"


def test_prediction_file_format(tmp_path, monkeypatch):
    # Train a tiny 1-epoch checkpoint on an in-memory split.
    mini = make_batch(4, seed=10, durations=(0, 4))
    ckpt_path = tmp_path / "snn_seed0.pt"
    cfg = TrainConfig(
        seed=0,
        data="generator",
        n_train=4,
        n_val=4,
        epochs=1,
        batch_size=4,
        device="cpu",
        checkpoint=str(ckpt_path),
        log_every=0,
    )
    train(cfg, train_split=mini, val_split=mini)

    # Patch iter_split in predict_snn so write_predictions runs quickly on 4 videos.
    monkeypatch.setattr("models.predict_snn.iter_split", lambda split, batch_size=64: [mini])
    snn_path, reset_path, every_path = write_predictions(
        seed=0,
        checkpoint=str(ckpt_path),
        out_dir=tmp_path,
        batch_size=4,
        device="cpu",
        reset_mode="both",
    )

    for path, expected_model in (
        (snn_path, "snn"),
        (reset_path, "snn_reset"),
        (every_path, "snn_reset_every"),
    ):
        saved = load_preds(path)
        assert saved["model"] == expected_model
        assert saved["seed"] == 0
        assert saved["data_tag"] == OFFICIAL_TEST_TAG
        assert int(saved["n_params"]) == EXPECTED_PARAMS
        assert saved["pred_xy"].shape == (4, 40, 2)
        assert saved["pred_xy"].dtype == np.float32
        assert np.isfinite(saved["pred_xy"]).all()
        assert (saved["pred_xy"] >= 0.0).all() and (saved["pred_xy"] <= 1.0).all()
        assert saved["spikes"].shape == (4, 40)
        assert saved["spikes"].dtype == np.float32
        assert (saved["spikes"] >= 0.0).all()

    # Also verify full (1000, 40, 2) compatibility with data.generator.iter_test_predictions.
    full_path = tmp_path / "snn_full_format_check.npz"
    save_preds(
        full_path,
        np.full((1000, 40, 2), 0.5, dtype=np.float32),
        model="snn",
        seed=0,
        data_tag=OFFICIAL_TEST_TAG,
        spikes=np.zeros((1000, 40), dtype=np.float32),
        n_params=EXPECTED_PARAMS,
    )
    rows = 0
    for pred_xy, true_xy, vis in iter_test_predictions(full_path, batch_size=250):
        assert pred_xy.shape == true_xy.shape
        assert vis.shape == true_xy.shape[:2]
        rows += len(pred_xy)
    assert rows == 1000
