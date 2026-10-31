#!/bin/bash
# Training run under Slurm (one node). ENV_PY (python) and DATA_DIR (tokenized ClimbMix) come in via --export;
# RUNS_ROOT (default ./runs) holds checkpoints/logs; NPROC = GPUs per node (default 4).
#   sbatch --gres=gpu:4 -c 16 --mem=48G -t 4-00:00:00 -J w3_sb_1b -o logs/%x-%j.out --signal=B:USR1@600 \
#          --export=ALL,ENV_PY=$HOME/env/bin/python,DATA_DIR=/data/climbmix slurm/w3_train_job.sh w3_sb_1b \
#          [extra train.py args, e.g. --set train.micro_batch=1]
# Resubmitting the same line resumes exactly from the last checkpoint (--resume auto).
set -uo pipefail
CFG=$1; shift
cd "${MORBIUS_DIR:-$HOME/morbius}"
: "${ENV_PY:?pass ENV_PY via --export}" "${DATA_DIR:?pass DATA_DIR via --export}"
[ -f "$DATA_DIR/index_seq2048_seed1234.npy" ] || { echo "no data/index at $DATA_DIR on $(hostname)"; exit 1; }
OUT=${RUNS_ROOT:-$PWD/runs}/$CFG
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
SCR=$(dirname "$(dirname "$OUT")")   # <scratch>/morbius
export WANDB_DIR="$OUT/logs" WANDB_CACHE_DIR=$SCR/wandb_cache WANDB_CONFIG_DIR=$SCR/wandb_config
echo "job ${SLURM_JOB_ID:-} $CFG on $(hostname) at $(date); out=$OUT data=$DATA_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$ENV_PY" scripts/print_params.py "configs/$CFG.yaml" || { echo "parameter check failed"; exit 1; }

# Slurm sends USR1 600 s before the time limit: ranks checkpoint at the next step and exit.
trap 'echo "USR1 at $(date)"; pkill -USR1 -P "$PID" || true; wait "$PID"; exit 0' USR1
"$ENV_PY" -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-4}" -m ffnmem.train \
  --config "configs/$CFG.yaml" --data_dir "$DATA_DIR" --out_dir "$OUT" --resume auto --wandb "$@" &
PID=$!
wait "$PID"
rc=$?
echo "exit $rc at $(date)"
exit $rc
