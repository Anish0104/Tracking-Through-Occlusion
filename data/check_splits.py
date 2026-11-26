"""Audit the three official splits with python -m data.check_splits."""

from collections import Counter
from hashlib import sha256

import numpy as np

from .generator import MARGIN, SPEED_BINS, SPLITS, iter_split
from .starter import SIZE


def audit():
    owners = {}
    for split, (size, _, durations) in SPLITS.items():
        counts = Counter()
        for batch in iter_split(split):
            for xy, frac, frames in zip(batch["xy"], batch["visible_frac"], batch["frames"]):
                path = sha256(xy.tobytes()).digest()
                if path in owners:
                    raise ValueError(f"Repeated trajectory in {owners[path]} and {split}")
                owners[path] = split
                hidden = np.flatnonzero(frac == 0)
                duration = len(hidden)
                if duration not in durations or (duration and hidden[-1] - hidden[0] + 1 != duration):
                    raise ValueError("Unexpected hidden interval")
                if np.any(frac[:MARGIN] != 1) or np.any(frac[-MARGIN:] != 1):
                    raise ValueError("Insufficient fully visible context")
                if duration == 0 and np.any(frac != 1):
                    raise ValueError("Occlusion in a fully visible control")
                if not np.array_equal((frames == 1).any(axis=(1, 2)), frac > 0):
                    raise ValueError("Rendered pixels disagree with visibility")
                dx = np.diff(xy[:, 0]) * SIZE
                if not (np.all(dx > 0) or np.all(dx < 0)):
                    raise ValueError("Direction reversal")
                speed = abs(dx.mean())
                matches = [i for i, (lo, hi) in enumerate(SPEED_BINS)
                           if lo - 1e-6 <= speed <= hi + 1e-6]
                if len(matches) != 1:
                    raise ValueError(f"Unexpected horizontal speed {speed}")
                counts[matches[0], duration] += 1
        table = [[counts[i, d] for d in durations] for i in range(len(SPEED_BINS))]
        if np.sum(table) != size or np.ptp(table) > 1:
            raise ValueError(f"Invalid split size or unbalanced cells in {split}")
        print(f"{split}: {size} videos; durations {durations}")
        for bounds, row in zip(SPEED_BINS, table):
            print(f"  {bounds} px/frame: {row}")
    print(f"{len(owners)} unique trajectories; zero overlap")


if __name__ == "__main__":
    audit()
