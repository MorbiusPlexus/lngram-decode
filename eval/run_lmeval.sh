#!/usr/bin/env bash
# Zero-shot lm-eval over the Wave 1-3 checkpoints (M6.T1), then the bootstrap table.
#
#   eval/run_lmeval.sh SB=runs/w3_sb_1b SD-B=runs/w3_sdb_1b          # NAME=CHECKPOINT_OR_RUN_DIR ...
#   CKPT_RUN1=... CKPT_RUN2=... CKPT_SB=... eval/run_lmeval.sh       # or from env (unset ones are skipped)
#
# Env: PY (python), DEVICE (cuda), BATCH (16), TASKS, OUT (results/lmeval), LIMIT, FORCE=1 (re-run done ones).
# MMLU is the cloze formulation: lm-eval group `mmlu_continuation` ("Question: ...\nAnswer:" + choice text).
# Per-sample outputs: $OUT/<name>/samples_<task>.jsonl; summary: results/lmeval_table.{json,md}.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
BATCH=${BATCH:-16}
TASKS=${TASKS:-hellaswag,mmlu_continuation,sciq,winogrande,piqa,arc_easy}
OUT=${OUT:-results/lmeval}
export TOKENIZERS_PARALLELISM=false

declare -a SPECS=()
if [ "$#" -gt 0 ]; then
  SPECS=("$@")
else
  # name -> env var holding the checkpoint (file or run dir)
  for pair in Run1:CKPT_RUN1 Run2:CKPT_RUN2 Run3:CKPT_RUN3 Run4:CKPT_RUN4 Run5:CKPT_RUN5 A:CKPT_A \
              SB:CKPT_SB SD-B:CKPT_SDB SD-C:CKPT_SDC SD-A:CKPT_SDA D270:CKPT_D270 BD:CKPT_BD \
              w2_mem_super1:CKPT_SUPER1; do
    name=${pair%%:*}; var=${pair#*:}
    if [ -n "${!var:-}" ]; then SPECS+=("$name=${!var}"); fi
  done
fi
if [ "${#SPECS[@]}" -eq 0 ]; then
  echo "no checkpoints: pass NAME=PATH args or set CKPT_* env vars" >&2
  exit 1
fi

mkdir -p "$OUT"
for spec in "${SPECS[@]}"; do
  name=${spec%%=*}; ckpt=${spec#*=}
  dir="$OUT/$name"
  if [ -f "$dir/results.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $name ($dir/results.json exists)"
    continue
  fi
  echo "[lmeval] $name <- $ckpt"
  extra=()
  if [ -n "${LIMIT:-}" ]; then extra+=(--limit "$LIMIT"); fi
  start=$(date +%s)
  "$PY" -m eval.lm_eval_adapter --checkpoint "$ckpt" --name "$name" --tasks "$TASKS" \
      --device "$DEVICE" --batch_size "$BATCH" --precision fp16 --num_fewshot 0 \
      --out_dir "$dir" ${extra[@]+"${extra[@]}"} 2>&1 | tee "$dir.log"
  echo "[lmeval] $name done in $(( $(date +%s) - start )) s"
done

"$PY" -m eval.bootstrap_lmeval --root "$OUT" --out results/lmeval_table.json
