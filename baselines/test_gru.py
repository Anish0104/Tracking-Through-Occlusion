"""Architecture tests for the CNN-GRU baseline (no full training). Run: python -m pytest -q"""

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from baselines import gru
from baselines.gru import (EXPECTED_PARAMS, SNN_PARAM_BUDGET, build_model, count_params,
                           param_breakdown, within_budget)
from common.loss import balanced_mse
from data.starter import make_batch

ATOL = 1e-6


@pytest.fixture(scope="module")
def model():
    return build_model(seed=0).eval()


@pytest.fixture(scope="module")
def batch():
    b = make_batch(n=6, seed=1)
    return {k: torch.from_numpy(v) for k, v in b.items()}


# ---------------------------------------------------------------- parameter budget

def test_exact_parameter_count(model):
    assert count_params(model) == EXPECTED_PARAMS == 1_067_064
    assert param_breakdown(model) == {"encoder": 4_800, "project": 966_774,
                                      "gru": 95_232, "readout": 258}


def test_within_ten_percent_of_snn_budget(model):
    n = count_params(model)
    assert SNN_PARAM_BUDGET == 1_070_274
    assert within_budget(n) and abs(n - SNN_PARAM_BUDGET) / SNN_PARAM_BUDGET < 0.005
    assert not within_budget(int(SNN_PARAM_BUDGET * 1.11))


def test_snn_reference_layout_matches_budget():
    # documents where 1,070,274 comes from (not an SNN implementation)
    ref = (nn.Conv2d(1, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1),
           nn.Linear(8192, 128), nn.Linear(128, 128), nn.Linear(128, 2))
    assert sum(count_params(m) for m in ref) == SNN_PARAM_BUDGET


def test_direct_gru_would_exceed_budget():
    assert not within_budget(4_800 + count_params(nn.GRU(8192, 128)) + 258)


# ---------------------------------------------------------------- architecture

def test_layer_layout(model):
    convs = [m for m in model.encoder if isinstance(m, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size) for c in convs] == [(1, 16, (3, 3)), (16, 32, (3, 3))]
    assert sum(isinstance(m, nn.MaxPool2d) for m in model.encoder) == 2
    assert model.project[0].in_features == 8192 and model.project[0].out_features == 118
    assert model.gru.input_size == 118 and model.gru.hidden_size == 128
    assert model.gru.num_layers == 1 and not model.gru.bidirectional
    assert (model.readout.in_features, model.readout.out_features) == (128, 2)
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_encoder_feature_size(model):
    assert model.encoder(torch.zeros(2, 1, 64, 64)).shape == (2, 8192)


def test_output_shape_range_dtype(model, batch):
    pred, h = model(batch["frames"])
    assert pred.shape == (6, 40, 2) and pred.dtype == torch.float32
    assert h.shape == (1, 6, 128)
    assert torch.isfinite(pred).all() and (pred > 0).all() and (pred < 1).all()


def test_rejects_wrong_shapes(model):
    with pytest.raises(ValueError):
        model(torch.zeros(2, 64, 64))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 3, 32, 32))
    with pytest.raises(ValueError):
        model.step(torch.zeros(2, 1, 64, 64))


# ---------------------------------------------------------------- causality and state

@torch.no_grad()
def test_step_by_step_matches_full_sequence(model, batch):
    frames = batch["frames"]
    full, h_full = model(frames)
    h, steps = None, []
    for t in range(frames.shape[1]):              # one frame at a time, in order
        xy, h = model.step(frames[:, t], h)
        steps.append(xy)
    assert torch.allclose(torch.stack(steps, 1), full, atol=ATOL)
    assert torch.allclose(h, h_full, atol=ATOL)


@torch.no_grad()
@pytest.mark.parametrize("t_cut", [0, 10, 25, 39])
def test_causal_future_frames_do_not_matter(model, batch, t_cut):
    frames = batch["frames"]
    noisy = frames.clone()
    noisy[:, t_cut + 1:] = torch.rand(noisy[:, t_cut + 1:].shape, generator=torch.Generator().manual_seed(t_cut))
    a, b = model(frames)[0], model(noisy)[0]
    assert torch.allclose(a[:, :t_cut + 1], b[:, :t_cut + 1], atol=ATOL)
    assert torch.allclose(model(frames[:, :t_cut + 1])[0], a[:, :t_cut + 1], atol=ATOL)


@torch.no_grad()
def test_videos_are_independent(model, batch):
    frames = batch["frames"]
    together = model(frames)[0]
    alone = torch.cat([model(frames[i:i + 1])[0] for i in range(len(frames))])
    assert torch.allclose(together, alone, atol=ATOL)
    perm = torch.randperm(len(frames), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(model(frames[perm])[0], together[perm], atol=ATOL)


@torch.no_grad()
def test_state_resets_between_videos(model, batch):
    frames = batch["frames"]
    assert torch.equal(model.init_hidden(3), torch.zeros(1, 3, 128))
    fresh = model(frames[1:2])[0]
    _, h_after_first = model(frames[0:1])
    carried = model(frames[1:2], h_after_first)[0]
    assert not torch.allclose(carried, fresh)         # carrying state would leak
    assert torch.allclose(model(frames[1:2])[0], fresh, atol=0)   # default = reset


# ---------------------------------------------------------------- reproducibility and gradients

def test_seeded_init_is_reproducible():
    a, b, c = build_model(seed=1), build_model(seed=1), build_model(seed=2)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_shared_loss_backprop_reaches_every_parameter(batch):
    m = build_model(seed=0).train()
    pred, _ = m(batch["frames"][:2, :10])
    loss = balanced_mse(pred, batch["xy"][:2, :10], batch["visible_frac"][:2, :10])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in m.parameters())


def test_few_optimizer_steps_reduce_loss_on_tiny_batch(batch):
    # sanity check only, not training: 2 short videos, 15 Adam steps
    m = build_model(seed=0).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x, y, v = batch["frames"][:2, :8], batch["xy"][:2, :8], batch["visible_frac"][:2, :8]
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = balanced_mse(m(x)[0], y, v)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


# ---------------------------------------------------------------- numpy inference helper

def test_predict_helper(model):
    frames = make_batch(n=5, seed=2)["frames"]
    assert list(inspect.signature(gru.predict).parameters)[:2] == ["frames", "model"]
    out = gru.predict(frames, model, batch_size=2)
    assert out.shape == (5, 40, 2) and out.dtype == np.float32 and np.isfinite(out).all()
    with torch.no_grad():
        ref = model(torch.from_numpy(frames))[0].numpy()
    assert np.allclose(out, ref, atol=ATOL)
    assert np.allclose(out, gru.predict(frames, model, batch_size=5), atol=ATOL)   # batch size irrelevant
    with pytest.raises(ValueError):
        gru.predict(frames[0], model)


def test_predict_restores_training_mode():
    m = build_model(seed=0).train()
    gru.predict(make_batch(n=1, seed=0)["frames"], m)
    assert m.training
