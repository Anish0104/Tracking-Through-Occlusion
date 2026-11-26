"""Last-observed-position baseline.

At each frame, output the detector's centroid if the disk is detected;
otherwise hold the most recent detection. Before the first detection in a
video, output the image center (0.5, 0.5). No motion model, no parameters,
no training. Frames are processed in order, one at a time, and each video
has its own state, so the prediction at frame t depends only on frames 0..t
of that video.

Partially visible frames count as detections (the centroid of the visible
part, see detector.py), so the held position during an occlusion is usually
the visible-part centroid from the frame just before the disk disappeared.
"""

import numpy as np

from baselines.detector import MIN_PIXELS, THRESHOLD, detect

START_XY = (0.5, 0.5)   # prediction before the first detection


def predict(frames, threshold=THRESHOLD, min_pixels=MIN_PIXELS):
    """frames: [N, T, 64, 64] video pixels only. Returns pred_xy float32 [N, T, 2]."""
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"frames must be [N, T, 64, 64], got shape {frames.shape}")
    n_videos, n_frames = frames.shape[:2]

    last = np.tile(np.asarray(START_XY, np.float32), (n_videos, 1))     # [N, 2] per-video state
    pred = np.empty((n_videos, n_frames, 2), np.float32)
    for t in range(n_frames):                      # causal: frame t only, in order
        xy, _, found = detect(frames[:, t], threshold, min_pixels)
        last = np.where(found[:, None], xy, last)
        pred[:, t] = last
    return pred
