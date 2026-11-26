"""Tests for eval/ and demo/. Run: python -m pytest -q

These check the metric definitions against hand-built cases where the right
answer is known by construction, rather than only checking that the code runs.
"""

import os

import numpy as np
import pytest

from common.io import save_preds
from data.starter import make_batch
from eval import stats
from eval.loader import (SNN_NEURONS, find_pred_files, load_labels, load_prediction,
                         load_all, make_fake_predictions, model_sort_key)
from eval.metrics import (SIZE, SUCCESS_PX, category_masks, has_occlusion, has_reappearance,
                          hidden_durations, per_video_mean, per_video_values, pixel_error,
                          reappearance_index, recovery_delay, spikes_per_neuron_per_frame,
                          summarize, summarize_by_duration)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def test_pixel_error_is_normalized_distance_times_size():
    true = np.zeros((2, 4, 2), np.float32)
    pred = true.copy()
    pred[..., 0] = 3.0 / SIZE                     # 3 px offset in x
    assert np.allclose(pixel_error(pred, true), 3.0)


def test_pixel_error_uses_both_axes():
    true = np.zeros((1, 1, 2))
    pred = np.array([[[3.0 / SIZE, 4.0 / SIZE]]])  # 3-4-5 triangle
    assert pixel_error(pred, true)[0, 0] == pytest.approx(5.0)


def test_pixel_error_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        pixel_error(np.zeros((2, 4, 2)), np.zeros((3, 4, 2)))


def test_categories_partition_every_frame():
    vf = np.array([[1.0, 0.5, 0.0, 1.0]])
    m = category_masks(vf)
    assert m["visible"].sum() + m["partial"].sum() + m["hidden"].sum() == vf.size
    assert m["overall"].all()
    assert list(m["hidden"][0]) == [False, False, True, False]


def test_per_video_mean_averages_videos_not_frames():
    # video 0 has 4 qualifying frames of error 1, video 1 has a single frame of 9.
    # Frame-weighted would give 2.6; per-video gives (1 + 9) / 2 = 5.
    values = np.array([[1.0, 1, 1, 1], [9.0, 0, 0, 0]])
    mask = np.array([[True] * 4, [True, False, False, False]])
    assert per_video_mean(values, mask) == (5.0, 2)


def test_per_video_mean_skips_videos_without_the_category():
    values = np.array([[4.0, 4.0], [99.0, 99.0]])
    mask = np.array([[True, True], [False, False]])
    mean, n = per_video_mean(values, mask)
    assert (mean, n) == (4.0, 1)                  # the second video must not count as 0


def test_per_video_mean_handles_no_qualifying_video():
    mean, n = per_video_mean(np.ones((2, 2)), np.zeros((2, 2), bool))
    assert np.isnan(mean) and n == 0


def test_per_video_values_marks_missing_videos_nan():
    vals = per_video_values(np.array([[2.0, 2.0], [5.0, 5.0]]),
                            np.array([[True, True], [False, False]]))
    assert vals[0] == 2.0 and np.isnan(vals[1])


def test_hidden_duration_counts_fully_hidden_frames():
    vf = np.array([[1.0, 0.0, 0.0, 0.3, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    assert list(hidden_durations(vf)) == [2, 0]


def test_hidden_durations_match_the_generator_request():
    from data.generator import make_batch as gen_batch
    for want in (0, 2, 4, 8, 12):
        b = gen_batch(6, 11, durations=(want,))
        assert set(hidden_durations(b["visible_frac"]).tolist()) == {want}


def test_reappearance_is_the_first_fully_visible_frame_after_hiding():
    vf = np.array([[1.0, 0.0, 0.0, 0.5, 1.0, 1.0]])
    assert reappearance_index(vf)[0] == 4         # the 0.5 frame is only partial
    assert reappearance_index(np.ones((1, 4)))[0] == -1          # never hidden
    assert reappearance_index(np.array([[1.0, 0.0, 0.0]]))[0] == -1   # ends hidden


def test_recovery_delay_counts_from_reappearance():
    vf = np.array([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    err = np.array([[0.0, 9, 9, 9, 1, 1, 1]])     # good from index 4 = offset 1
    delay, recovered = recovery_delay(err, vf)
    assert recovered[0] and delay[0] == 1


def test_recovery_delay_is_zero_when_already_accurate():
    vf = np.array([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    err = np.array([[0.0, 9, 9, 1, 1, 1, 1]])
    assert recovery_delay(err, vf)[0][0] == 0


def test_recovery_needs_three_consecutive_good_frames():
    vf = np.array([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    err = np.array([[0.0, 9, 9, 1, 9, 1, 1]])     # 1, then a miss, then only 2 good
    delay, recovered = recovery_delay(err, vf)
    assert not recovered[0] and np.isnan(delay[0])


def test_recovery_marks_never_recovered_when_error_stays_high():
    vf = np.array([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    err = np.full((1, 7), 20.0)
    delay, recovered = recovery_delay(err, vf)
    assert not recovered[0] and np.isnan(delay[0])


def test_videos_that_never_reappear_are_excluded_from_recovery():
    # A video that ends while hidden, and one that reappears too late to show
    # three good frames, are both unfair to score: a perfect tracker would fail.
    ends_hidden = np.array([[1.0, 1.0, 0.0, 0.0]])
    short_tail = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])      # only 2 frames after
    assert not has_reappearance(ends_hidden)[0]
    assert not has_reappearance(short_tail)[0]
    assert has_reappearance(np.array([[1.0, 0.0, 1.0, 1.0, 1.0]]))[0]


def test_perfect_tracker_never_fails_to_recover():
    """The never-recovered rate must measure the model, not the data."""
    b = make_batch(n=60, seed=5)
    vf = b["visible_frac"]
    s = summarize(b["xy"], b["xy"], vf)           # exact predictions
    assert s["recovery"]["never_recovered"] == 0.0
    assert s["recovery"]["delay_mean"] == 0.0


def test_summarize_is_zero_for_perfect_predictions():
    b = make_batch(n=8, seed=4)
    s = summarize(b["xy"], b["xy"], b["visible_frac"])
    for cat in ("overall", "visible", "hidden"):
        if s["categories"][cat]["videos"]:
            assert s["categories"][cat]["err_px"] == pytest.approx(0.0)
            assert s["categories"][cat]["success"] == pytest.approx(1.0)


def test_summarize_success_threshold_is_five_pixels():
    vf = np.ones((2, 3))
    true = np.zeros((2, 3, 2))
    pred = true.copy()
    pred[0, :, 0] = (SUCCESS_PX - 0.1) / SIZE     # just inside
    pred[1, :, 0] = (SUCCESS_PX + 0.1) / SIZE     # just outside
    s = summarize(pred, true, vf)
    assert s["categories"]["overall"]["success"] == pytest.approx(0.5)


def test_summarize_by_duration_splits_videos():
    b = make_batch(n=40, seed=6)
    out = summarize_by_duration(b["xy"], b["xy"], b["visible_frac"])
    assert set(out) == set(hidden_durations(b["visible_frac"]).tolist())
    assert sum(v["n_videos"] for v in out.values()) == 40


def test_spikes_per_neuron_per_frame():
    spikes = np.full((2, 5), 1000.0)
    assert spikes_per_neuron_per_frame(spikes, 100) == pytest.approx(10.0)
    assert np.isnan(spikes_per_neuron_per_frame(None, 100))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    v = rng.normal(5.0, 1.0, 300)
    mean, lo, hi, n = stats.bootstrap_mean_ci(v, n_boot=800)
    assert lo < mean < hi and n == 300
    assert stats.bootstrap_mean_ci(v, n_boot=800) == (mean, lo, hi, n)


def test_bootstrap_ci_ignores_nans():
    v = np.array([1.0, 1.0, 1.0, np.nan])
    mean, lo, hi, n = stats.bootstrap_mean_ci(v, n_boot=200)
    assert n == 3 and mean == pytest.approx(1.0)


def test_bootstrap_ci_handles_empty_input():
    mean, lo, hi, n = stats.bootstrap_mean_ci([])
    assert n == 0 and np.isnan(mean)


def test_paired_diff_detects_a_consistent_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(10, 3, 300)                 # shared per-video difficulty
    a = base + rng.normal(0, 0.1, 300)
    b = base + 0.5 + rng.normal(0, 0.1, 300)
    r = stats.paired_diff_ci(a, b, n_boot=800)
    assert r["mean_diff"] == pytest.approx(-0.5, abs=0.05)
    assert r["significant"] and r["high"] < 0
    assert r["frac_a_better"] > 0.9


def test_paired_diff_of_identical_values_is_not_significant():
    v = np.random.default_rng(2).normal(0, 1, 100)
    r = stats.paired_diff_ci(v, v.copy(), n_boot=400)
    assert r["mean_diff"] == 0.0 and not r["significant"]


def test_paired_diff_keeps_pairs_aligned():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([2.0, np.nan, 3.0, 5.0])
    assert stats.paired_diff_ci(a, b, n_boot=100)["n_pairs"] == 2


def test_paired_diff_requires_matching_shapes():
    with pytest.raises(ValueError):
        stats.paired_diff_ci(np.zeros(3), np.zeros(4))


def test_mean_sd_uses_the_sample_spread():
    mean, sd, n = stats.mean_sd([1.0, 2.0, 3.0])
    assert (mean, n) == (2.0, 3) and sd == pytest.approx(1.0)
    assert stats.mean_sd([7.0]) == (7.0, 0.0, 1)


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #

def test_model_sort_order_puts_known_models_first():
    assert sorted(["last_pos", "zzz", "snn"], key=model_sort_key) == ["snn", "last_pos", "zzz"]


def test_fake_predictions_round_trip_and_score(tmp_path):
    labels = load_labels("starter", n=12, seed=300)
    make_fake_predictions(labels, models=("snn", "last_pos"), seeds=(0,), out_dir=str(tmp_path))
    labels2, entries, problems = load_all(str(tmp_path), source="starter", n=12, seed=300)
    assert not problems and len(entries) == 2
    models = {e["model"] for e in entries}
    assert models == {"snn", "last_pos"}
    snn = next(e for e in entries if e["model"] == "snn")
    assert snn["spikes"] is not None and snn["n_params"] == 1_070_274
    # the placeholder SNN is built to be better than last position while hidden
    hid = {e["model"]: summarize(e["pred_xy"], labels2["xy"],
                                 labels2["visible_frac"])["categories"]["hidden"]["err_px"]
           for e in entries}
    assert hid["snn"] < hid["last_pos"]


def test_loader_refuses_predictions_for_a_different_number_of_videos(tmp_path):
    labels = load_labels("starter", n=12, seed=300)
    make_fake_predictions(labels, models=("snn",), seeds=(0,), out_dir=str(tmp_path))
    path = os.path.join(str(tmp_path), "snn_seed0.npz")
    with pytest.raises(ValueError, match="pred_xy is"):
        load_prediction(path, load_labels("starter", n=6, seed=300))


def test_loader_refuses_a_mismatched_data_tag(tmp_path):
    labels = load_labels("starter", n=12, seed=300)
    make_fake_predictions(labels, models=("snn",), seeds=(0,), out_dir=str(tmp_path))
    path = os.path.join(str(tmp_path), "snn_seed0.npz")
    other = load_labels("starter", n=12, seed=301)
    with pytest.raises(ValueError, match="data_tag"):
        load_prediction(path, other)
    assert load_prediction(path, other, strict_tag=False)["warning"]


def test_load_all_reports_bad_files_instead_of_crashing(tmp_path):
    labels = load_labels("starter", n=12, seed=300)
    make_fake_predictions(labels, models=("snn",), seeds=(0,), out_dir=str(tmp_path))
    save_preds(os.path.join(str(tmp_path), "broken_seed0.npz"),
               np.zeros((3, 40, 2), np.float32), model="broken", seed=0,
               data_tag=labels["tag"])
    _, entries, problems = load_all(str(tmp_path), source="starter", n=12, seed=300)
    assert [e["model"] for e in entries] == ["snn"]
    assert len(problems) == 1 and "broken" in problems[0][0]


def test_find_pred_files_sorts_by_model_then_seed(tmp_path):
    labels = load_labels("starter", n=6, seed=300)
    make_fake_predictions(labels, models=("gru", "snn"), seeds=(0, 1), out_dir=str(tmp_path))
    names = [os.path.basename(p) for p in find_pred_files(str(tmp_path))]
    assert names == ["snn_seed0.npz", "snn_seed1.npz", "gru_seed0.npz", "gru_seed1.npz"]


def test_snn_neuron_count_matches_the_architecture():
    assert SNN_NEURONS == 16 * 64 * 64 + 32 * 32 * 32 + 128


# --------------------------------------------------------------------------- #
# report and figures
# --------------------------------------------------------------------------- #

def test_report_runs_and_covers_every_required_section(tmp_path):
    from eval.report import run
    labels = load_labels("starter", n=40, seed=300)
    make_fake_predictions(labels, out_dir=str(tmp_path))
    text, results = run(str(tmp_path), source="starter", n=40, seed=300, n_boot=200)
    for heading in ("visibility category", "hidden duration", "recovery",
                    "bootstrap", "paired", "motion", "compute"):
        assert heading in text
    assert results["n_videos"] == 40
    assert set(results["comparisons"]) >= {"hidden", "overall"}
    assert results["by_duration"]


def test_report_aggregates_seeds_without_double_counting(tmp_path):
    from eval.report import build_runs, group_by_model, seed_averaged_per_video
    labels = load_labels("starter", n=20, seed=300)
    make_fake_predictions(labels, models=("gru",), seeds=(0, 1, 2), out_dir=str(tmp_path))
    _, entries, _ = load_all(str(tmp_path), source="starter", n=20, seed=300)
    data = build_runs(labels, entries)
    by_model = group_by_model(data["runs"])
    assert len(by_model["gru"]) == 3
    assert seed_averaged_per_video(by_model["gru"], "overall").shape == (20,)


def test_report_refuses_when_there_is_nothing_to_score(tmp_path):
    from eval.report import run
    with pytest.raises(SystemExit):
        run(str(tmp_path), source="starter", n=10, seed=300)


def test_figures_are_written(tmp_path):
    from eval.figures import make_all
    preds = tmp_path / "preds"
    figs = tmp_path / "figures"
    labels = load_labels("starter", n=30, seed=300)
    make_fake_predictions(labels, out_dir=str(preds))
    written, problems = make_all(str(preds), "starter", 30, 300, str(figs), n_boot=100)
    assert not problems and len(written) >= 4
    for path in written:
        assert os.path.getsize(path) > 1000


def test_animation_is_written(tmp_path):
    from demo.animate import animate, pick_videos
    from eval.report import build_runs, group_by_model
    preds = tmp_path / "preds"
    labels = load_labels("starter", n=20, seed=300)
    make_fake_predictions(labels, models=("snn", "last_pos"), seeds=(0,), out_dir=str(preds))
    _, entries, _ = load_all(str(preds), source="starter", n=20, seed=300)
    data = build_runs(labels, entries)
    for r, e in zip(data["runs"], entries):
        r["pred"] = e["pred_xy"]
    by_model = group_by_model(data["runs"])
    videos = pick_videos(labels, by_model, "worst", 2)
    assert len(videos) == 2 and all(isinstance(v, int) for v in videos)
    out = animate(labels, by_model, videos, str(tmp_path / "demo.gif"), fps=4)
    assert os.path.getsize(out) > 1000
