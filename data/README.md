# data/ (owner: Krisha)

`starter.py` is a small stand-in so the others can start today. Your job is `generator.py`, which replaces it with the same output format (CONTRACT.md, section 1).

To deliver:
- `generator.py` with `make_batch(n, seed, ...)`: random barrier width and position, constant velocity first, smooth velocity changes later
- control over fully hidden durations 2, 4, 8, 12 plus zero-occlusion videos; balanced speed-vs-duration bins by adjusting barrier geometry
- reject videos without enough visible frames before and after occlusion; no bounces while hidden
- fixed train / val / test seeds (100 / 200 / 300) and a check that no trajectory appears in two splits
- `check_splits.py` and a short note on the generator settings
