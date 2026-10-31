#!/bin/bash
# One node, 2 GPUs (e.g. 2x 16 GB): no ZeRO, micro-batch 4 (accum 32); MICRO_BATCH=2 for SD models.
#   DATA_DIR=/data/climbmix PY=~/env/bin/python nohup scripts/launch_2gpu.sh w3_sdc_1b > runs/w3_sdc_1b.log 2>&1 &
# A run must resume with the same zero1 setting it started with (ZeRO-1 is off here).
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
$PY -m torch.distributed.run --standalone --nproc_per_node=${NPROC:-2} -m ffnmem.train \
  --config "configs/$CFG.yaml" --data_dir "$DATA_DIR" --out_dir "$OUT" --resume auto --wandb \
  --set train.micro_batch=${MICRO_BATCH:-4} --set train.zero1=false "$@" &
PID=$!
wait "$PID"; rc=$?
echo "exit $rc at $(date)"; exit $rc
