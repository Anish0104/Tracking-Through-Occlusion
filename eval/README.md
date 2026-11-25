# eval/ and demo/ (owner: Kunal)

Build everything on `common/fake.py` predictions now; real prediction files plug in later with no changes.

To deliver:
- `metrics.py`: center error in pixels on fully hidden / fully visible / partial frames (mean per video, then over videos); success rate at 5 px; recovery delay and never-recovered fraction; spikes per neuron per frame
- `report.py`: reads every file in `preds/`, groups by hidden duration (0, 2, 4, 8, 12), reports per-seed results, mean and spread, paired per-video comparisons and bootstrap confidence intervals
- figures: error vs hidden duration, recovery delay vs hidden duration, predicted vs true trajectories
- compute table: parameters, training time, inference time on one machine
- `demo/animate.py`: side-by-side animation of the scene, each model's marker and the true path; include failure cases
