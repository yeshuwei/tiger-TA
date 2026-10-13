#!/usr/bin/env bash
# =============================================================================
# run_clean_tiger.sh — entry point for the frozen Clean TIGER baseline
# =============================================================================
# Runs the original TIGER pipeline end to end and nothing else:
#
#   item content representation -> RQ-VAE -> Semantic ID -> T5 -> Recommendation
#
# Usage:
#   bash clean_tiger/run_clean_tiger.sh [stage]
#
#   stage = rqvae | sid | t5 | eval | verify | all      (default: all)
#
# Environment overrides (see config.sh for the full list):
#   DATASET=Beauty SEED_T5=2026 bash clean_tiger/run_clean_tiger.sh all
#
# All outputs go under clean_tiger/ckpt/ and clean_tiger/logs/, so this never
# touches the artifacts of the historical experiments.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${HERE}/config.sh"

STAGE="${1:-all}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$RQVAE_CKPT_DIR" "$CLEAN_LOG_DIR"

log()  { echo "[$(date '+%F %T')] [clean-tiger] $*"; }
fail() { echo "✗ [clean-tiger] $*" >&2; exit 1; }

banner() {
  echo "=================================================================="
  echo "[clean-tiger] stage=${STAGE}  dataset=${DATASET}"
  echo "  content emb : ${CONTENT_EMB}"
  echo "  rqvae ckpt  : ${RQVAE_CKPT_DIR}"
  echo "  sid output  : ${SID_OUT}"
  echo "  t5 ckpt     : ${T5_CKPT}"
  echo "  historical switches: lambda_trans=${LAMBDA_TRANS} lambda_balance=${LAMBDA_BALANCE} align_weight=${ALIGN_WEIGHT}"
  echo "=================================================================="
}

# -----------------------------------------------------------------------------
# Stage 1 — RQ-VAE on the content-only item embedding
# -----------------------------------------------------------------------------
stage_rqvae() {
  assert_clean_emb "$CONTENT_EMB"
  log "RQ-VAE training on content-only embedding (epochs=${RQVAE_EPOCHS}, seed=${SEED_RQVAE})"
  # rqvae/main.py declares --bn / --kmeans_init with `type=bool`, so argparse
  # turns the STRING "False" into True (non-empty string is truthy). Passing
  # "--bn False" would therefore silently enable batch norm, which original
  # TIGER does not use. Only pass these flags when they are actually True and
  # otherwise let the script's own default (bn=False, kmeans_init=True) stand.
  local bool_args=()
  if [ "$RQVAE_BN" = "True" ]; then bool_args+=(--bn True); fi
  if [ "$RQVAE_KMEANS_INIT" = "True" ]; then bool_args+=(--kmeans_init True); fi
  cd "${ROOT}/rqvae"
  python main.py \
    --data_path        "$CONTENT_EMB" \
    --ckpt_dir         "$RQVAE_CKPT_DIR" \
    --seed             "$SEED_RQVAE" \
    --lr               "$RQVAE_LR" \
    --epochs           "$RQVAE_EPOCHS" \
    --batch_size       "$RQVAE_BATCH" \
    --num_workers      "$RQVAE_NUM_WORKERS" \
    --eval_step        "$RQVAE_EVAL_STEP" \
    --learner          "$RQVAE_LEARNER" \
    --lr_scheduler_type "$RQVAE_LR_SCHEDULER" \
    --warmup_epochs    "$RQVAE_WARMUP_EPOCHS" \
    --weight_decay     "$RQVAE_WEIGHT_DECAY" \
    --dropout_prob     "$RQVAE_DROPOUT" \
    --loss_type        "$RQVAE_LOSS_TYPE" \
    ${bool_args[@]+"${bool_args[@]}"} \
    --kmeans_iters     "$RQVAE_KMEANS_ITERS" \
    --sk_iters         "$RQVAE_SK_ITERS" \
    --beta             "$RQVAE_BETA" \
    --quant_loss_weight "$RQVAE_QUANT_LOSS_WEIGHT" \
    --save_limit       "$RQVAE_SAVE_LIMIT" \
    --num_emb_list     "${RQVAE_NUM_EMB_LIST[@]}" \
    --e_dim            "$RQVAE_E_DIM" \
    --layers           "${RQVAE_LAYERS[@]}" \
    --sk_epsilons      "${RQVAE_SK_EPS[@]}" \
    --device           "$DEVICE" \
    $(rqvae_clean_args) \
    2>&1 | tee "${CLEAN_LOG_DIR}/rqvae_${DATASET}_seed${SEED_RQVAE}.log"
}

# -----------------------------------------------------------------------------
# Stage 2 — Semantic ID generation from the trained RQ-VAE
# -----------------------------------------------------------------------------
stage_sid() {
  local ckpt="${RQVAE_CKPT:-$(newest_rqvae_ckpt "$RQVAE_CKPT_DIR")}"
  [ -n "$ckpt" ] && [ -f "$ckpt" ] || fail "no RQ-VAE checkpoint under ${RQVAE_CKPT_DIR} (run stage 'rqvae' first)"
  log "Generating Semantic IDs"
  log "  from ${ckpt}"
  log "  to   ${SID_OUT}"
  cd "${ROOT}/rqvae"
  python generate_code.py \
    --dataset     "$DATASET" \
    --ckpt_path   "$ckpt" \
    --output_file "$SID_OUT" \
    --device      "$DEVICE" \
    2>&1 | tee "${CLEAN_LOG_DIR}/sid_${DATASET}.log"
}

# -----------------------------------------------------------------------------
# Stage 3 — T5 recommender on the Semantic IDs
# -----------------------------------------------------------------------------
stage_t5() {
  [ -f "$SID_OUT" ] || fail "Semantic ID file not found: ${SID_OUT} (run stage 'sid' first)"
  log "T5 training (epochs=${T5_NUM_EPOCHS}, seed=${SEED_T5})"
  cd "${ROOT}"
  python model/main.py \
    --dataset_path       "${DATA_ROOT}/${DATASET}" \
    --code_path          "$SID_OUT" \
    --save_path          "$T5_CKPT" \
    --log_path           "$T5_LOG" \
    --seed               "$SEED_T5" \
    --batch_size         "$T5_BATCH" \
    --infer_size         "$T5_INFER_SIZE" \
    --num_epochs         "$T5_NUM_EPOCHS" \
    --lr                 "$T5_LR" \
    --num_layers         "$T5_NUM_LAYERS" \
    --num_decoder_layers "$T5_NUM_DECODER_LAYERS" \
    --d_model            "$T5_D_MODEL" \
    --d_ff               "$T5_D_FF" \
    --num_heads          "$T5_NUM_HEADS" \
    --d_kv               "$T5_D_KV" \
    --dropout_rate       "$T5_DROPOUT" \
    --max_len            "$T5_MAX_LEN" \
    --beam_size          "$T5_BEAM" \
    --early_stop         "$T5_EARLY_STOP" \
    2>&1 | tee "${CLEAN_LOG_DIR}/t5_${DATASET}_seed${SEED_T5}_train.log"
}

# -----------------------------------------------------------------------------
# Stage 4 — evaluation of the trained T5
# -----------------------------------------------------------------------------
stage_eval() {
  [ -f "$T5_CKPT" ] || fail "T5 checkpoint not found: ${T5_CKPT} (run stage 't5' first)"
  log "Evaluating ${T5_CKPT}"
  cd "${ROOT}"
  python3 tools/eval_trained_model.py \
    --dataset    "$DATASET" \
    --ckpt_path  "$T5_CKPT" \
    --code_path  "$SID_OUT" \
    --beam_size  "$T5_BEAM" \
    --topk_list  5,10,20,50,100 \
    --seed       "$SEED_T5" \
    2>&1 | tee "${CLEAN_LOG_DIR}/t5_${DATASET}_seed${SEED_T5}_eval.log"
}

# -----------------------------------------------------------------------------
# Stage 5 — prove that no historical mechanism participates
# -----------------------------------------------------------------------------
stage_verify() {
  log "Verifying that the Clean TIGER RQ-VAE path loads no historical mechanism"
  cd "${ROOT}"
  python -I clean_tiger/verify_clean.py \
    --root "$ROOT" \
    --data_path "$CONTENT_EMB" \
    --device "$DEVICE"
}

banner
case "$STAGE" in
  rqvae)  stage_rqvae ;;
  sid)    stage_sid ;;
  t5)     stage_t5 ;;
  eval)   stage_eval ;;
  verify) stage_verify ;;
  all)    stage_rqvae; stage_sid; stage_t5; stage_eval ;;
  *)      fail "unknown stage '${STAGE}' (rqvae|sid|t5|eval|verify|all)" ;;
esac
log "stage '${STAGE}' done"
