#!/bin/bash
# SSM gate on 2 GPUs: post-hoc drafters on frozen SB without / with SSM (one GPU each, in parallel),
# offline acceptance on long-document prompts, then the gate (results/gates.json).
#   sbatch --gres=gpu:2 -c 8 --mem=24G -t 1-00:00:00 -J w3_ssm_gate -o logs/%x-%j.out \
#          --export=ALL,ENV_PY=$HOME/env/bin/python,DATA_DIR=/data/climbmix,RUNS_ROOT=$PWD/runs slurm/w3_ssm_gate.sh
# Resumable: distillations continue (--resume auto) and generated records are reused.
set -uo pipefail
cd "${MORBIUS_DIR:-$HOME/morbius}"
: "${ENV_PY:?pass ENV_PY via --export}" "${DATA_DIR:?pass DATA_DIR via --export}"
R=${RUNS_ROOT:-$PWD/runs}
SB=${SB_RUN:-$R/w3_sb_1b}
TOKENS=${PH_TOKENS:-200e6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "ssm gate on $(hostname) at $(date); target $SB"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for v in "ph_sb 0" "ph_sb_ssm 256"; do
  set -- $v
  gpu=$([ "$2" = 0 ] && echo 0 || echo 1)
  CUDA_VISIBLE_DEVICES=$gpu "$ENV_PY" -m drafting.posthoc_distill --target "$SB" --data_dir "$DATA_DIR" \
    --out_dir "$R/$1" --ssm_dim "$2" --tokens "$TOKENS" --resume auto --wandb > "$R/$1.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] || { echo "a distillation failed; see $R/ph_sb*.log"; exit 1; }

CUDA_VISIBLE_DEVICES=0 "$ENV_PY" -m drafting.offline_acceptance --target "$SB" --name w3_sb_1b --data_dir "$DATA_DIR" \
  --types long --drafters pld "posthoc=posthoc@$R/ph_sb/drafter.pt" "posthoc_ssm=posthoc@$R/ph_sb_ssm/drafter.pt" \
  --cache_dir "$R/acceptance_cache/w3_sb_1b" || exit 1
"$ENV_PY" -m drafting.gates --ssm_file results/acceptance_w3_sb_1b.json \
  --acceptance results/acceptance_w3_sb_1b.json --out results/gates.json
echo "done at $(date)"
