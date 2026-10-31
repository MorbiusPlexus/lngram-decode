#!/bin/bash
# Download + tokenize ClimbMix, build the fixed training order, verify (CPU only, resumable).
#   sbatch -c 8 --mem=24G -t 1-00:00:00 -J w3_data -o logs/%x-%j.out \
#          --export=ALL,ENV_PY=$HOME/env/bin/python,DATA_DIR=/data/climbmix slurm/w3_data_job.sh
# 2.2B train tokens covers every run (the dense control needs 3,815 steps x 524,288 = 2.0B).
set -euo pipefail
cd "${MORBIUS_DIR:-$HOME/morbius}"
: "${ENV_PY:?pass ENV_PY via --export}"
DATA_DIR=${DATA_DIR:-$PWD/data/climbmix}
TOKENS=${TOKENS:-2200000000}
export HF_HOME=${HF_HOME:-$(dirname "$DATA_DIR")/hf_home} PYTHONUNBUFFERED=1   # keep HF caches off the home quota
mkdir -p "$DATA_DIR" "$HF_HOME"
echo "data job on $(hostname) at $(date): $TOKENS tokens -> $DATA_DIR"
RESUME=""
[ -f "$DATA_DIR/meta.json" ] && RESUME="--resume"
"$ENV_PY" data/download_dataset.py --tokenizer data/tokenizer --output_dir "$DATA_DIR" \
  --target_tokens "$TOKENS" --val_tokens 50000000 $RESUME
[ -f "$DATA_DIR/index_seq2048_seed1234.npy" ] || "$ENV_PY" -m ffnmem.data.build_index --data "$DATA_DIR"
"$ENV_PY" data/verify_dataset.py --data_dir "$DATA_DIR" --max_shards 2 --n_samples 3
echo "done at $(date)"; ls -la "$DATA_DIR"; du -sh "$DATA_DIR"
