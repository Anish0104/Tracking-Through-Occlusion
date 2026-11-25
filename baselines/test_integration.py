"""Integration tests: switching generators, provenance checks, data_tag, Kalman retuning,
the contract checker and the multi-seed shell wrapper. Seed 300 is never generated.
Run: python -m pytest -q"""

import json
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pytest

from baselines import check_data, data_cache, dev_metrics, predict as predict_mod, tune_kalman
from baselines.train_gru import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "baselines" / "train_gru_seeds.sh"

# A contract-shaped stand-in for Krisha's generator: durations exactly 0/2/4/8,
# public iter_videos hook, sequential RNG stream. Pixel levels follow CONTRACT.md.
FAKE_GENERATOR = """
    import numpy as np
    T = 40

    def _video(rng):
        d = int(rng.choice([0, 2, 4, 8]))
        start = int(rng.integers(10, 25))
        vis = np.ones(T, np.float32)
        vis[start:start + d] = 0.0
        x0, y0 = rng.uniform(0.2, 0.3), rng.uniform(0.3, 0.7)
        t = np.arange(T, dtype=np.float32)
        xy = np.stack([x0 + 0.01 * t, np.full(T, y0)], -1).astype(np.float32)
        frames = np.zeros((T, 64, 64), np.float32)
        frames[:, :, 30:34] = 0.4
        for i in range(T):
            if vis[i] > 0:
                c, r = int(xy[i, 0] * 64), int(xy[i, 1] * 64)
                frames[i, r - 1:r + 2, c - 1:c + 2] = 1.0
        return frames, xy, vis

    def make_batch(n, seed, **kwargs):
        rng = np.random.default_rng(seed)
        out = [_video(rng) for _ in range(n)]
        return {"frames": np.stack([o[0] for o in out]), "xy": np.stack([o[1] for o in out]),
                "visible_frac": np.stack([o[2] for o in out])}

    def iter_videos(n, seed):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            f, xy, v = _video(rng)
            yield {"frames": f, "xy": xy, "visible_frac": v}
"""


@pytest.fixture
def fakegen(tmp_path, monkeypatch):
    name = f"fakegen_{uuid.uuid4().hex[:8]}"
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(FAKE_GENERATOR))
    monkeypatch.syspath_prepend(str(tmp_path))
    return name, path


@pytest.fixture
def no_test_bank(monkeypatch):
    """Fail loudly if anything asks any generator (or the starter adapter) for seed 300."""
    real = data_cache.resolve_module

    def guard(fn):
        def wrapped(*args, **kw):
            seed = kw.get("seed", args[1] if len(args) > 1 else None)
            assert seed != 300, "test bank generated"
            return fn(*args, **kw)
        return wrapped

    def resolve(source):
        mod = real(source)
        for attr in ("make_batch", "iter_videos"):
            if hasattr(mod, attr) and not getattr(getattr(mod, attr), "_guarded", False):
                g = guard(getattr(mod, attr))
                g._guarded = True
                monkeypatch.setattr(mod, attr, g)
        return mod
    monkeypatch.setattr(data_cache, "resolve_module", resolve)
    adapter = data_cache.ADAPTERS["data.starter"]
    monkeypatch.setitem(data_cache.ADAPTERS, "data.starter",
                        lambda m, n, seed: guard(adapter)(m, n, seed=seed))


def _train_tiny(tmp_path, data, seed=0, **kw):
    ckpt = tmp_path / f"{data.replace('.', '_')}_gru_seed{seed}.pt"
    cfg = dict(seed=seed, data=data, n_train=6, n_val=3, epochs=1, batch_size=3, device="cpu",
               log_every=0, checkpoint=str(ckpt))
    cfg.update(kw)
    train(TrainConfig(**cfg))
    return ckpt


# ---------------------------------------------------------------- generator switch

def test_missing_generator_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "data.generator", None)
    with pytest.raises(ModuleNotFoundError, match="has not been merged yet"):
        data_cache.resolve_module("generator")


def test_switching_generator_trains_and_uses_separate_cache(tmp_path, fakegen):
    name, _ = fakegen
    cache = tmp_path / "cache"
    s = _train_tiny(tmp_path, "starter", cache_dir=str(cache))
    g = _train_tiny(tmp_path, name, cache_dir=str(cache))
    dirs = sorted(p.name for p in cache.iterdir())
    assert len(dirs) == 4                                     # 2 splits x 2 generators, no reuse
    assert sum(d.startswith("data_starter_") for d in dirs) == 2
    assert sum(d.startswith(name) for d in dirs) == 2
    import torch
    cs = torch.load(s, weights_only=True)["config"]
    cg = torch.load(g, weights_only=True)["config"]
    assert (cs["data_module"], cg["data_module"]) == ("data.starter", name)
    assert cs["data_fingerprint"] != cg["data_fingerprint"]


def test_fake_final_generator_passes_contract_check_and_starter_fails_durations(fakegen, no_test_bank):
    name, _ = fakegen
    res = {r[1]: r[0] for r in check_data.check(name, n=30)}
    assert "FAIL" not in res.values(), res
    starter = {r[1]: r[0] for r in check_data.check("starter", n=60)}
    assert starter["hidden durations"] == "FAIL"
    assert [k for k, v in starter.items() if v == "FAIL"] == ["hidden durations"]


def test_hidden_runs():
    v = np.array([1, 0, 0, 0.5, 0, 1, 0])
    assert check_data.hidden_runs(v) == [2, 1, 1]
    assert check_data.hidden_runs(np.ones(5)) == []


# ---------------------------------------------------------------- provenance

def test_gru_prediction_refuses_other_generator(tmp_path, fakegen):
    name, _ = fakegen
    ckpt = _train_tiny(tmp_path, "starter")
    kw = dict(model="gru", n=3, data_seed=200, seed=0, out_dir=tmp_path / "p", checkpoint=str(ckpt))
    with pytest.raises(ValueError, match="fitted on data.starter"):
        predict_mod.run(data=name, **kw)
    out = predict_mod.run(data=name, allow_data_mismatch=True, **kw)    # explicit override
    assert out.exists()
    predict_mod.run(data="data.starter", **kw)                          # alias of the same source


def test_gru_prediction_refuses_edited_generator(tmp_path, fakegen):
    name, path = fakegen
    ckpt = _train_tiny(tmp_path, name)
    path.write_text(path.read_text() + "\n# changed after training\n")
    with pytest.raises(ValueError, match="has changed"):
        predict_mod.run(model="gru", data=name, n=2, data_seed=200, out_dir=tmp_path,
                        checkpoint=str(ckpt))


def test_kalman_config_provenance(fakegen, monkeypatch, tmp_path):
    name, _ = fakegen
    assert predict_mod._kalman_provenance()[0] == "data.starter"   # committed config (old format)
    predict_mod._predictor("kalman", data="starter")
    with pytest.raises(ValueError, match="kalman_config.json"):
        predict_mod._predictor("kalman", data=name)
    predict_mod._predictor("kalman", data=name, allow_data_mismatch=True)


def test_data_tag_is_alias_independent():
    assert predict_mod.default_data_tag("starter", 300) == "starter_test_s300"
    assert predict_mod.default_data_tag("data.starter", 300) == "starter_test_s300"


def test_all_baselines_share_one_bank_and_tag(tmp_path, monkeypatch):
    calls = []
    real = predict_mod.run.__globals__["save_preds"]
    monkeypatch.setitem(predict_mod.run.__globals__, "save_preds",
                        lambda path, pred, **kw: calls.append(kw["data_tag"]) or real(path, pred, **kw))
    ckpt = _train_tiny(tmp_path, "starter")
    for model in ("last_pos", "kalman"):
        predict_mod.run(model=model, data="starter", n=3, data_seed=200, out_dir=tmp_path)
    predict_mod.run(model="gru", data="data.starter", n=3, data_seed=200, out_dir=tmp_path,
                    checkpoint=str(ckpt))
    assert calls == ["starter_test_s200"] * 3


# ---------------------------------------------------------------- Kalman retuning

def test_kalman_retune_on_new_generator_validation_only(fakegen, tmp_path, monkeypatch, capsys,
                                                       no_test_bank):
    name, _ = fakegen
    out = tmp_path / "kalman_config.json"
    monkeypatch.setattr(tune_kalman, "CONFIG_PATH", out)
    monkeypatch.setattr(tune_kalman, "GRID", {"accel_px": [0.01, 0.1], "meas_px": [0.2],
                                              "partial_px": [0.0, 4.0], "min_pixels": [1]})
    tune_kalman.main(["--data", name, "--n", "6", "--write"])
    doc = json.loads(out.read_text())
    assert doc["selection"]["module"] == name and doc["selection"]["data_seed"] == 200
    assert len(doc["selection"]["fingerprint"]) == 64
    assert "WARNING: accel_px=" in capsys.readouterr().out    # 2-value grid: best is at an edge


def test_edge_warnings():
    from baselines.kalman import KalmanConfig
    grid = {"accel_px": [0.001, 0.01, 0.1], "meas_px": [0.2, 0.4]}
    w = tune_kalman.edge_warnings(KalmanConfig(accel_px=0.001, meas_px=0.2), grid)
    assert any("accel_px=0.001 is at the lower edge" in x for x in w)
    assert tune_kalman.edge_warnings(KalmanConfig(accel_px=0.01, meas_px=0.3), grid) == []


def test_dev_metrics_accepts_data_and_mismatch_override(fakegen, capsys, no_test_bank):
    name, _ = fakegen
    with pytest.raises(ValueError):
        dev_metrics.main(["--data", name, "--n", "3", "--models", "kalman"])
    dev_metrics.main(["--data", name, "--n", "3", "--models", "last_pos", "kalman",
                      "--allow-data-mismatch"])
    assert f"{name} data seed 200" in capsys.readouterr().out


# ---------------------------------------------------------------- shell wrapper exit codes

def _run_script(tmp_path, env_extra, *args):
    env = {**os.environ, "OUT_DIR": str(tmp_path / "ck"), "CACHE_DIR": str(tmp_path / "cache"),
           "N_TRAIN": "6", "N_VAL": "3", "EPOCHS": "1", "DEVICE": "cpu", "SEEDS": "0 1",
           "PYTHON": sys.executable, **env_extra}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


def test_tee_alone_hides_failures():
    # Why the wrapper uses pipefail: a plain pipeline reports tee's exit status.
    r = subprocess.run(["bash", "-c", "false | tee /dev/null"])
    assert r.returncode == 0
    r = subprocess.run(["bash", "-c", "set -o pipefail; false | tee /dev/null"])
    assert r.returncode != 0


def test_wrapper_trains_all_seeds(tmp_path):
    r = _run_script(tmp_path, {})
    assert r.returncode == 0, r.stderr + r.stdout
    for s in (0, 1):
        assert (tmp_path / "ck" / f"gru_seed{s}.pt").exists()
        assert (tmp_path / "ck" / f"gru_seed{s}.json").exists()
        assert "epoch  1" in (tmp_path / "ck" / f"gru_seed{s}.log").read_text()
    r = _run_script(tmp_path, {})                              # second run: refuses to overwrite
    assert r.returncode == 2 and "refusing" in r.stderr


def test_wrapper_stops_on_training_failure(tmp_path):
    fake = tmp_path / "fakepython"
    fake.write_text(f"""#!/usr/bin/env bash
if [[ "$*" == *baselines.train_gru* ]]; then echo "boom" >&2; exit 7; fi
exec {sys.executable} "$@"
""")
    fake.chmod(0o755)
    r = _run_script(tmp_path, {"PYTHON": str(fake)})
    assert r.returncode == 7                                   # the real failure code, not tee's 0
    assert not (tmp_path / "ck" / "gru_seed1.log").exists()    # seed 1 never started
    assert "boom" in (tmp_path / "ck" / "gru_seed0.log").read_text()


def test_wrapper_rejects_run_without_summary(tmp_path):
    fake = tmp_path / "fakepython"
    fake.write_text(f"""#!/usr/bin/env bash
if [[ "$*" == *baselines.train_gru* ]]; then echo "exited 0 but wrote nothing"; exit 0; fi
exec {sys.executable} "$@"
""")
    fake.chmod(0o755)
    r = _run_script(tmp_path, {"PYTHON": str(fake)})
    assert r.returncode == 1 and "did not write a fresh summary" in r.stderr
