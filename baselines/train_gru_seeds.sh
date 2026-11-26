#!/usr/bin/env bash
# Train the CNN-GRU for several model seeds, one after another, on one shared data cache.
#
#   DATA=generator bash baselines/train_gru_seeds.sh            # final runs (seeds 0 1 2, MPS)
#   DATA=starter N_TRAIN=200 N_VAL=50 bash baselines/train_gru_seeds.sh   # small starter dry run
#
# Final-data guard: DATA=starter (or data.starter) with N_TRAIN >= 5000 is refused before
# anything is cached or trained, so the starter stand-in cannot be used for the final
# experiment by accident. ALLOW_STARTER=1 overrides this for a deliberate experiment.
#
# Failure handling: `set -o pipefail` makes `cmd | tee log` fail when cmd fails (without
# it the pipeline reports tee's success), `set -e` stops at the first failing seed, and the
# script's exit code is that failure's. A seed only counts as done when train_gru wrote its
# .json summary during this run (the summary is written after the last epoch).
# Existing checkpoints are never replaced unless OVERWRITE=1.
#
# Settings (environment variables; defaults are the CONTRACT.md / Phase 4 values):
#   DATA=starter DEVICE=mps SEEDS="0 1 2" N_TRAIN=5000 N_VAL=1000 EPOCHS=30 PATIENCE=5
#   BATCH_SIZE=16 LR=1e-3 CACHE_DIR=checkpoints/data_cache OUT_DIR=checkpoints PYTHON=python
# Extra arguments are passed to every train_gru call.
set -euo pipefail

DATA="${DATA:-starter}"
DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-0 1 2}"
N_TRAIN="${N_TRAIN:-5000}"
N_VAL="${N_VAL:-1000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
CACHE_DIR="${CACHE_DIR:-checkpoints/data_cache}"
OUT_DIR="${OUT_DIR:-checkpoints}"
PYTHON="${PYTHON:-python}"
OVERWRITE="${OVERWRITE:-0}"
ALLOW_STARTER="${ALLOW_STARTER:-0}"

STARTER_FLAG=()
if [[ "$DATA" == starter || "$DATA" == data.starter ]] && (( N_TRAIN >= 5000 )); then
  if [[ "$ALLOW_STARTER" != 1 ]]; then
    echo "refusing to start: DATA=$DATA with N_TRAIN=$N_TRAIN is a full-size run on the starter" \
         "data; final runs use DATA=generator (ALLOW_STARTER=1 overrides)" >&2
    exit 2
  fi
  STARTER_FLAG=(--allow-starter)
fi

mkdir -p "$OUT_DIR"
for s in $SEEDS; do
  if [[ -e "$OUT_DIR/gru_seed$s.pt" && "$OVERWRITE" != 1 ]]; then
    echo "refusing to start: $OUT_DIR/gru_seed$s.pt exists (set OVERWRITE=1 to replace it)" >&2
    exit 2
  fi
done

# Build (or checksum-verify) the shared train/val cache once; seed 300 is never cached.
"$PYTHON" -m baselines.data_cache --data "$DATA" --cache-dir "$CACHE_DIR" \
  --n-train "$N_TRAIN" --n-val "$N_VAL" 2>&1 | tee "$OUT_DIR/data_cache.log"

for s in $SEEDS; do
  started="$(mktemp)"
  "$PYTHON" -m baselines.train_gru --seed "$s" --data "$DATA" \
    --n-train "$N_TRAIN" --n-val "$N_VAL" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --lr "$LR" --patience "$PATIENCE" --device "$DEVICE" --cache-dir "$CACHE_DIR" \
    --checkpoint "$OUT_DIR/gru_seed$s.pt" --log-every 0 ${STARTER_FLAG[@]+"${STARTER_FLAG[@]}"} "$@" \
    2>&1 | tee "$OUT_DIR/gru_seed$s.log"
  if [[ ! "$OUT_DIR/gru_seed$s.json" -nt "$started" ]]; then
    rm -f "$started"
    echo "seed $s: training did not write a fresh summary; treating as failed" >&2
    exit 1
  fi
  rm -f "$started"
  echo "seed $s: done -> $OUT_DIR/gru_seed$s.pt"
done
