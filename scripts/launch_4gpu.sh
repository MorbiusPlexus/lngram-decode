#!/bin/bash
# One node, 4 GPUs (e.g. 4x 11 GB): ZeRO-1, micro-batch 2 (accum 32). Resumes automatically.
#   DATA_DIR=/data/climbmix PY=~/env/bin/python nohup scripts/launch_4gpu.sh w3_sb_1b > runs/w3_sb_1b.log 2>&1 &
#   Out of memory (SD models on 11 GB cards): add --set train.micro_batch=1
set -uo pipefail
CFG=$1; shift
cd "${MORBIUS_DIR:-$(dirname "$0")/..}"
PY=${PY:-python}
DATA_DIR=${DATA_DIR:?set DATA_DIR to the tokenized ClimbMix dir (with index_seq2048_seed1234.npy)}
OUT=${RUNS_ROOT:-runs}/$CFG
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
export WANDB_PROJECT=${WANDB_PROJECT:-morbius-wave3} WANDB_DIR="$OUT/logs"
echo "$CFG on $(hostname) at $(date); out=$OUT data=$DATA_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
$PY scripts/print_params.py "configs/$CFG.yaml" || { echo "parameter check failed"; exit 1; }
trap 'pkill -USR1 -P "$PID" || true; wait "$PID"; exit 0' USR1 TERM
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $PY -m torch.distributed.run --standalone --nproc_per_node=4 -m ffnmem.train \
  --config "configs/$CFG.yaml" --data_dir "$DATA_DIR" --out_dir "$OUT" --resume auto --zero1 --wandb \
  --set train.micro_batch=2 "$@" &
PID=$!
wait "$PID"; rc=$?
echo "exit $rc at $(date)"; exit $rc
