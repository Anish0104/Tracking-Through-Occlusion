# Tracking Through Occlusion

Team 15, Brain-Inspired Computing (16:198:525), Fall 2026.
Anuj Bohra, Kunal Pasad, Anish V Shirodkar, Krisha Borana.

Does a recurrent spiking neural network keep track of a moving object while the object is hidden behind a barrier, and does its internal state help it recover when the object reappears? We compare it with a CNN-GRU, a Kalman tracker and a last-position baseline on generated videos with exact labels.

## Who owns what

| Folder | Owner | Task |
|---|---|---|
| `data/` | Krisha | Full video generator, labels, splits |
| `models/` | Anish | Recurrent SNN, training, state-reset ablation, spike counts |
| `baselines/` | Anuj | CNN-GRU, centroid detector + Kalman, last-position |
| `eval/`, `demo/` | Kunal | Metrics, statistics, figures, animation |
| `common/` | Everyone | Shared loss and prediction-file format (change only by agreement) |

The rules that keep these tasks independent are in [CONTRACT.md](CONTRACT.md). The full task description is in `docs/Team15_Task_Division.pdf`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q              # should print "4 passed"
```

## Quick look at the data

```python
from data.starter import make_batch
b = make_batch(n=4, seed=0)
print(b["frames"].shape, b["xy"].shape, b["visible_frac"].shape)
```
