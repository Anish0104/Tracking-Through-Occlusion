"""Exactness and robustness tests for baselines/data_cache.py (tiny splits, seconds).
Run: python -m pytest -q"""

import json
import os
import subprocess
import sys
import textwrap
import uuid

import numpy as np
import pytest
import torch

from baselines import data_cache, train_gru
from baselines.data_cache import (CacheError, PaletteArray, PaletteEncoder, build_split,
                                  get_split, open_split)
from baselines.train_gru import TrainConfig, iterate_batches, load_split, train
from data import starter
from data.starter import make_batch

KEYS = data_cache.DATA_KEYS
N = 12


def _bits_equal(a, b):
    return data_cache._bitwise_equal(np.asarray(a), np.asarray(b))


def _assert_identical(cached, ref):
    for k in KEYS:
        got = np.asarray(cached[k])
        assert got.dtype == ref[k].dtype, k
        assert got.shape == ref[k].shape, k
        assert _bits_equal(got, ref[k]), k


def _quiet(*a, **k):
    pass


def _fake_module(tmp_path, monkeypatch, body):
    """Write a generator module to a real file (so it has a source fingerprint)."""
    name = f"fakegen_{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


# A contract-shaped generator with configurable distinct pixel values. It has a
# sequential RNG stream like the starter, and optionally the public iter_videos hook.
FAKE_GEN = """
    import numpy as np
    LEVELS = {levels}

    def _video(rng, i):
        if LEVELS is None or i >= {switch_at}:
            frames = rng.random((5, 64, 64), dtype=np.float32)       # thousands of values
        else:
            frames = rng.choice(np.array(LEVELS, np.float32), size=(5, 64, 64))
        xy = rng.random((5, 2), dtype=np.float32)
        vis = rng.random(5, dtype=np.float32)
        return frames, xy, vis

    def make_batch(n, seed):
        rng = np.random.default_rng(seed)
        out = [_video(rng, i) for i in range(n)]
        return {{"frames": np.stack([o[0] for o in out]), "xy": np.stack([o[1] for o in out]),
                "visible_frac": np.stack([o[2] for o in out])}}
"""
ITER_HOOK = """
    def iter_videos(n, seed):
        rng = np.random.default_rng(seed)
        for i in range(n):
            f, xy, v = _video(rng, i)
            yield {"frames": f, "xy": xy, "visible_frac": v}
"""


# ---------------------------------------------------------------- 1-3: exact equality, order

@pytest.mark.parametrize("fmt", ["auto", "uint8", "raw"])
def test_cached_starter_split_is_bitwise_identical(tmp_path, fmt):
    ref = make_batch(N, 100)
    cached = get_split("starter", N, 100, tmp_path, fmt=fmt, log=_quiet)
    _assert_identical(cached, ref)
    manifest = json.loads(next(tmp_path.glob("data_starter_n12_s100_*/manifest.json")).read_text())
    assert manifest["producer"] == "adapter:data.starter"         # incremental path used
    expected = "raw" if fmt == "raw" else "palette_u8"
    assert manifest["arrays"]["frames"]["encoding"] == expected
    assert manifest["arrays"]["xy"]["encoding"] == "raw"


def test_uint8_palette_is_lossless_and_small(tmp_path):
    cached = get_split("starter", N, 200, tmp_path, log=_quiet)
    assert isinstance(cached["frames"], PaletteArray)
    assert cached["frames"].codes.dtype == np.uint8 and isinstance(cached["frames"].codes, np.memmap)
    assert sorted(cached["frames"].palette.tolist()) == [0.0, np.float32(0.4), 1.0]
    assert _bits_equal(np.asarray(cached["frames"]), make_batch(N, 200)["frames"])
    d = next(tmp_path.glob("data_starter_*"))
    assert (d / "frames_codes.npy").stat().st_size < N * 40 * 64 * 64 + 1024   # 1 byte / pixel


def test_palette_encoder_preserves_bit_patterns():
    x = np.array([0.0, -0.0, np.nan, 0.4, 1.0, np.float32(0.4)], np.float32)
    enc = PaletteEncoder(x.dtype)
    codes = enc.encode(x)
    assert len(enc.values) == 5                              # -0.0 and NaN kept distinct
    assert _bits_equal(enc.values[codes], x)
    assert enc.encode(np.arange(300, dtype=np.float32)) is None      # > 256 values refused


def test_sample_order_preserved_for_arbitrary_index_order(tmp_path):
    ref = make_batch(N, 100)
    cached = get_split("starter", N, 100, tmp_path, log=_quiet)
    idx = np.array([7, 2, 11, 0, 2, 5])
    for k in KEYS:
        assert _bits_equal(train_gru._take(cached[k], idx), ref[k][idx]), k
    for i in (0, 5, N - 1):                                   # video i is make_batch's video i
        assert _bits_equal(cached["frames"][i], ref["frames"][i])


# ---------------------------------------------------------------- 4: seeds reproduce data

def test_same_seed_reproduces_identical_cache_files(tmp_path):
    a = build_split("starter", N, 100, tmp_path / "a", log=_quiet)
    b = build_split("starter", N, 100, tmp_path / "b", log=_quiet)
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in (a, b))
    assert a.name == b.name
    assert {k: v["sha256"] for k, v in ma["arrays"].items()} == \
           {k: v["sha256"] for k, v in mb["arrays"].items()}
    c = get_split("starter", N, 101, tmp_path / "a", log=_quiet)
    assert not np.array_equal(np.asarray(c["frames"]), make_batch(N, 100)["frames"])


def test_train_and_val_caches_are_distinct_and_test_seed_refused(tmp_path):
    build_split("starter", N, 100, tmp_path, log=_quiet)
    build_split("starter", N, 200, tmp_path, log=_quiet)
    assert len(list(tmp_path.glob("data_starter_*"))) == 2
    with pytest.raises(CacheError):
        build_split("starter", N, 300, tmp_path, log=_quiet)
    assert not list(tmp_path.glob("*s300*"))


# ---------------------------------------------------------------- 5: shared across model seeds

def test_model_seeds_share_one_cached_dataset(tmp_path, monkeypatch):
    builds = []
    real = data_cache._write_split
    monkeypatch.setattr(data_cache, "_write_split",
                        lambda *a, **k: builds.append(1) or real(*a, **k))
    cfg = dict(n_train=8, n_val=4, epochs=1, batch_size=4, device="cpu", log_every=0,
               cache_dir=str(tmp_path / "cache"))
    r0 = train(TrainConfig(seed=0, checkpoint=str(tmp_path / "s0.pt"), **cfg))
    r1 = train(TrainConfig(seed=1, checkpoint=str(tmp_path / "s1.pt"), **cfg))
    assert len(builds) == 2                                   # train + val, built once in total
    assert len(list((tmp_path / "cache").iterdir())) == 2
    assert r0["data_fingerprint"] == r1["data_fingerprint"]
    assert r0["history"][0]["train_loss"] != r1["history"][0]["train_loss"]   # different models


# ---------------------------------------------------------------- 6-7: batches and training

def test_batches_from_cache_equal_in_memory_batches(tmp_path):
    mem = load_split("starter", N, 100)
    for fmt in ("auto", "raw"):
        cached = load_split("starter", N, 100, cache_dir=tmp_path / fmt, cache_format=fmt)
        order = np.random.default_rng(3).permutation(N)
        for bm, bc in zip(iterate_batches(mem, 5, order), iterate_batches(cached, 5, order)):
            for tm, tc in zip(bm, bc):
                assert tm.dtype == tc.dtype == torch.float32
                assert tm.device.type == "cpu" and len(tm) <= 5      # one batch at a time, on CPU
                assert torch.equal(tm, tc)


def test_training_with_cache_matches_in_memory_training(tmp_path):
    cfg = dict(seed=0, n_train=8, n_val=4, epochs=2, batch_size=3, device="cpu", log_every=0)
    mem = train(TrainConfig(checkpoint=str(tmp_path / "mem.pt"), **cfg))
    cached = train(TrainConfig(checkpoint=str(tmp_path / "c.pt"), cache_dir=str(tmp_path / "cache"), **cfg))
    assert [(h["train_loss"], h["val_loss"]) for h in mem["history"]] == \
           [(h["train_loss"], h["val_loss"]) for h in cached["history"]]
    a = torch.load(tmp_path / "mem.pt", weights_only=True)["model_state"]
    b = torch.load(tmp_path / "c.pt", weights_only=True)["model_state"]
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------- 8: interruption and corruption

def test_interrupted_build_leaves_no_valid_cache_and_is_rebuilt(tmp_path, monkeypatch):
    def broken(module, n, seed):
        for i, v in enumerate(data_cache._iter_starter(module, n, seed)):
            if i == 6 and n == N:
                raise KeyboardInterrupt                       # e.g. Ctrl-C mid-build
            yield v

    monkeypatch.setitem(data_cache.ADAPTERS, "data.starter", broken)
    with pytest.raises(KeyboardInterrupt):
        build_split("starter", N, 100, tmp_path, log=_quiet)
    assert list(tmp_path.iterdir()) == []                     # nothing that looks like a cache
    monkeypatch.setitem(data_cache.ADAPTERS, "data.starter", data_cache._iter_starter)
    _assert_identical(get_split("starter", N, 100, tmp_path, log=_quiet), make_batch(N, 100))


def test_killed_build_partial_dir_is_ignored_then_removed(tmp_path):
    name = data_cache.split_name(starter, N, 100)
    proc = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True)
    dead = int(proc.stdout)                                   # a pid that has exited
    partial = tmp_path / f".{name}.partial-{dead}-deadbeef"
    partial.mkdir()
    (partial / "frames_codes.npy").write_bytes(b"half written")
    alive = tmp_path / f".{name}.partial-{os.getpid()}-cafecafe"   # a live build: left alone
    alive.mkdir()
    with pytest.raises(CacheError):
        open_split(partial)                                   # never valid without a manifest
    msgs = []
    path = build_split("starter", N, 100, tmp_path, log=msgs.append)
    assert not partial.exists() and alive.exists()
    assert any("removed interrupted" in m for m in msgs)
    _assert_identical(open_split(path), make_batch(N, 100))


def test_corrupted_cache_is_rejected(tmp_path):
    path = build_split("starter", N, 100, tmp_path, log=_quiet)
    f = path / "xy.npy"
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF                                          # flip one byte
    f.write_bytes(bytes(data))
    with pytest.raises(CacheError, match="checksum"):
        open_split(path)
    (path / "manifest.json").write_text('{"format_version": 1, "complete": false}')
    with pytest.raises(CacheError):
        open_split(path)


# ---------------------------------------------------------------- other generators

def test_generator_with_public_iter_videos_hook(tmp_path, monkeypatch):
    name = _fake_module(tmp_path, monkeypatch,
                        FAKE_GEN.format(levels="[0.0, 0.25, 0.7]", switch_at=10**9) + ITER_HOOK)
    mod = data_cache.resolve_module(name)
    cached = get_split(name, 9, 100, tmp_path / "c", log=_quiet)
    _assert_identical(cached, mod.make_batch(9, 100))
    m = json.loads(next((tmp_path / "c").glob("*/manifest.json")).read_text())
    assert m["producer"] == "iter_videos" and m["arrays"]["frames"]["encoding"] == "palette_u8"


def test_generator_without_hook_falls_back_to_make_batch_and_keeps_dtype(tmp_path, monkeypatch):
    name = _fake_module(tmp_path, monkeypatch, FAKE_GEN.format(levels="None", switch_at=0))
    mod = data_cache.resolve_module(name)
    cached = get_split(name, 7, 100, tmp_path / "c", log=_quiet)
    _assert_identical(cached, mod.make_batch(7, 100))
    m = json.loads(next((tmp_path / "c").glob("*/manifest.json")).read_text())
    assert m["producer"] == "make_batch"
    assert m["arrays"]["frames"]["encoding"] == "raw"          # > 256 values: original dtype
    with pytest.raises(CacheError, match="lossy"):
        build_split(name, 7, 200, tmp_path / "u", fmt="uint8", log=_quiet)
    assert not [p for p in (tmp_path / "u").iterdir() if not p.name.startswith(".")]


def test_palette_overflow_midway_switches_to_raw_exactly(tmp_path, monkeypatch):
    name = _fake_module(tmp_path, monkeypatch,
                        FAKE_GEN.format(levels="[0.0, 0.5, 1.0]", switch_at=4) + ITER_HOOK)
    mod = data_cache.resolve_module(name)
    path = build_split(name, 8, 100, tmp_path / "c", log=_quiet)
    assert not (path / "frames_codes.npy").exists()
    _assert_identical(open_split(path), mod.make_batch(8, 100))


def test_broken_incremental_producer_is_caught_before_caching(tmp_path, monkeypatch):
    def wrong_stream(module, n, seed):
        return data_cache._iter_starter(module, n, seed + 1)

    monkeypatch.setitem(data_cache.ADAPTERS, "data.starter", wrong_stream)
    with pytest.raises(CacheError, match="disagrees"):
        build_split("starter", N, 100, tmp_path, log=_quiet)
    assert not list(tmp_path.glob("data_*"))


def test_generator_edit_gives_new_cache_and_keeps_old(tmp_path, monkeypatch):
    name = _fake_module(tmp_path, monkeypatch, FAKE_GEN.format(levels="[0.0, 1.0]", switch_at=10**9))
    first = build_split(name, 4, 100, tmp_path / "c", log=_quiet)
    src = tmp_path / f"{name}.py"
    src.write_text(src.read_text() + "\n# edited\n")
    second = build_split(name, 4, 100, tmp_path / "c", log=_quiet)
    assert first != second and first.exists() and second.exists()


def test_data_cache_cli_builds_and_fully_verifies(tmp_path, capsys):
    data_cache.main(["--cache-dir", str(tmp_path), "--n-train", "6", "--n-val", "3", "--verify-full"])
    out = capsys.readouterr().out
    assert out.count("bitwise equal to make_batch") == 2 and "False" not in out
    assert "(6, 100)" in out and "(3, 200)" in out
