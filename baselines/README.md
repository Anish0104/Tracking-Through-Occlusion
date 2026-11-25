# baselines/ (owner: Anuj)

To deliver:
- `gru.py` + `train_gru.py`: convolutional encoder + GRU + the same coordinate readout; parameter count within 10% of 1,070,274 (report the exact number); same loss, epoch cap and tuning budget as the SNN; seeds 0, 1, 2
- `kalman.py`: centroid detector on pixels only (threshold above the barrier brightness of 0.4; document how partial visibility is handled) + constant-velocity Kalman filter that predicts without a measurement on hidden frames; tune noise on validation data
- `last_pos.py`: hold the last visible estimate through occlusion
- prediction files for `gru`, `kalman`, `last_pos`
