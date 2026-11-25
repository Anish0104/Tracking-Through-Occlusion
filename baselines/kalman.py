"""Constant-velocity Kalman filter on top of the pixel-only centroid detector.

Measurements come from baselines/detector.py; the tracker never sees labels.
Whether a frame has a measurement is decided from pixels only (the detector's
bright-pixel count), never from `visible_frac`.

State and model (all in normalized units, i.e. pixels / 64; dt = 1 frame)
    s = [x, y, vx, vy]
    predict:  s <- F s,              P <- F P F^T + Q
    update:   S = H P H^T + R,       K = P H^T S^-1
              s <- s + K (z - H s),  P <- (I - K H) P   (then symmetrised)
    F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],  H = [[1,0,0,0],[0,1,0,0]]
    Q = sigma_a^2 G G^T with G = [[1/2,0],[0,1/2],[1,0],[0,1]]
        (white random acceleration of std `accel_px` px/frame^2 per axis)

Per frame, per video
    no detection yet        output the image center (0.5, 0.5), no state
    first detection         s = [z, 0, 0];  P = diag(R, R, v0^2, v0^2), v0 = init_vel_px
    later frames            predict; then update only if there is a usable detection
    no usable detection     prediction only (this is how occlusion is bridged)

Partial visibility (confidence from the pixel count)
    The barrier is drawn over the disk, so a partly hidden disk gives the
    centroid of its visible part, biased toward the uncovered side by up to
    about one radius. We treat that bias as extra measurement noise that grows
    as fewer pixels are visible:
        w = min(1, n_pixels / full_px)                   (visible fraction estimate)
        R = (meas_px^2 + (partial_px * (1 - w))^2) / 64^2 * I
    so a fully visible disk has std `meas_px` and a sliver approaches
    sqrt(meas_px^2 + partial_px^2). partial_px = 0 trusts every detection
    equally. Detections with fewer than `min_pixels` bright pixels are ignored
    (treated as no measurement).

Clipping policy
    The internal state is never clipped (velocity must stay consistent). Only
    the reported position is clipped to [0, 1], the image extent, so a long
    extrapolation cannot point outside the frame. All other outputs are the
    filter's posterior position.

Hyperparameters are loaded from baselines/kalman_config.json, produced by
baselines/tune_kalman.py on the validation bank (data seed 200).
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from baselines.detector import SIZE, THRESHOLD, detect

CONFIG_PATH = Path(__file__).with_name("kalman_config.json")
START_XY = (0.5, 0.5)

F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], np.float64)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]], np.float64)
_G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], np.float64)


@dataclass(frozen=True)
class KalmanConfig:
    accel_px: float = 0.03       # process noise: random-acceleration std, px / frame^2
    meas_px: float = 0.3         # measurement noise std for a fully visible disk, px
    partial_px: float = 2.0      # extra std added as visibility goes to zero, px
    full_px: float = 28.0        # bright pixels of a fully visible disk (pi * 3^2 ~ 28)
    min_pixels: int = 1          # detections with fewer bright pixels are ignored
    init_vel_px: float = 2.0     # initial velocity std at the first detection, px / frame
    threshold: float = THRESHOLD # detector threshold (above barrier brightness 0.4)

    def __post_init__(self):
        for name in ("accel_px", "meas_px", "full_px", "init_vel_px"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.partial_px < 0:
            raise ValueError("partial_px must be >= 0")
        if self.min_pixels < 1:
            raise ValueError("min_pixels must be >= 1")


def load_config(path=CONFIG_PATH):
    """Read the tuned config; fields not in the file keep their defaults."""
    with open(path) as f:
        params = json.load(f)["params"]
    known = {f.name for f in fields(KalmanConfig)}
    return KalmanConfig(**{k: v for k, v in params.items() if k in known})


def default_config():
    return load_config() if CONFIG_PATH.exists() else KalmanConfig()


def measurement_std_px(n_pixels, config):
    """Measurement noise std (pixels) for detections with `n_pixels` bright pixels."""
    w = np.clip(np.asarray(n_pixels, np.float64) / config.full_px, 0.0, 1.0)
    return np.sqrt(config.meas_px ** 2 + (config.partial_px * (1.0 - w)) ** 2)


def process_noise(config):
    sa = config.accel_px / SIZE
    return sa ** 2 * _G @ _G.T


def track(xy, n_pixels, found, config=None, return_state=False):
    """Run the filter on detector outputs.

    xy [N, T, 2], n_pixels [N, T], found [N, T] as returned by detector.detect.
    Returns pred_xy float32 [N, T, 2]. With return_state=True also returns
    a dict with the posterior `state` [N, T, 4], `cov` [N, T, 4, 4],
    `updated` [N, T] (measurement used) and `active` [N, T] (filter started).
    """
    config = config or default_config()
    xy = np.asarray(xy, np.float64)
    n_pixels = np.asarray(n_pixels)
    found = np.asarray(found, bool)
    if xy.ndim != 3 or xy.shape[-1] != 2 or n_pixels.shape != xy.shape[:2] or found.shape != xy.shape[:2]:
        raise ValueError("expected xy [N, T, 2], n_pixels [N, T], found [N, T]")
    N, T = found.shape

    use = found & (n_pixels >= config.min_pixels)            # usable measurement, from pixels only
    r_var = (measurement_std_px(n_pixels, config) / SIZE) ** 2  # [N, T]
    Q = process_noise(config)
    v0 = (config.init_vel_px / SIZE) ** 2

    s = np.zeros((N, 4))                      # per-video state: nothing shared across videos
    P = np.zeros((N, 4, 4))
    active = np.zeros(N, bool)
    pred = np.empty((N, T, 2), np.float32)
    if return_state:
        states, covs = np.zeros((N, T, 4)), np.zeros((N, T, 4, 4))
        updated, actives = np.zeros((N, T), bool), np.zeros((N, T), bool)

    for t in range(T):                        # causal: frame t uses only detections 0..t
        z, u, rv = xy[:, t], use[:, t], r_var[:, t]

        # 1. predict every running filter
        a = active
        s[a] = s[a] @ F.T
        P[a] = F @ P[a] @ F.T + Q

        # 2. update running filters that have a usable measurement
        m = a & u
        if m.any():
            Pm = P[m]
            S = Pm[:, :2, :2] + rv[m, None, None] * np.eye(2)
            K = Pm[:, :, :2] @ np.linalg.inv(S)                    # [k, 4, 2]
            innov = z[m] - s[m, :2]
            s[m] = s[m] + np.einsum("kij,kj->ki", K, innov)
            Pm = (np.eye(4) - K @ H) @ Pm
            P[m] = 0.5 * (Pm + Pm.transpose(0, 2, 1))

        # 3. start filters at their first usable measurement
        new = ~a & u
        if new.any():
            s[new] = np.concatenate([z[new], np.zeros((new.sum(), 2))], axis=1)
            P[new] = np.diag([0, 0, v0, v0])[None].repeat(new.sum(), 0)
            P[new, 0, 0] = P[new, 1, 1] = rv[new]
            active = active | new

        out = np.where(active[:, None], s[:, :2], np.asarray(START_XY))
        pred[:, t] = np.clip(out, 0.0, 1.0)
        if return_state:
            states[:, t], covs[:, t] = s, P
            updated[:, t], actives[:, t] = u & active, active

    if return_state:
        return pred, {"state": states, "cov": covs, "updated": updated, "active": actives}
    return pred


def predict(frames, config=None):
    """frames: [N, T, 64, 64] video pixels only. Returns pred_xy float32 [N, T, 2]."""
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"frames must be [N, T, 64, 64], got shape {frames.shape}")
    config = config or default_config()
    xy, n, found = detect(frames, threshold=config.threshold)
    return track(xy, n, found, config)


def config_dict(config):
    return asdict(config)
