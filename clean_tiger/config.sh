#!/usr/bin/env bash
# =============================================================================
# clean_tiger/config.sh — frozen Clean TIGER baseline configuration
# =============================================================================
# Single source of truth for the Clean TIGER pipeline. Sourced by
# run_clean_tiger.sh, smoke_test.sh and (indirectly) verify_clean.py.
#
# Clean TIGER = the original TIGER pipeline, and nothing else:
#
#   item content representation  ->  RQ-VAE  ->  Semantic ID  ->  T5  ->  Rec
#
# Every historical experiment switch is pinned to 0 here and is passed
# EXPLICITLY on the command line (never left to a default that a future edit
# could change). Historical code is untouched and still available for the
# existing experiments — see the "excluded" block below.
#
# Override any value from the environment, e.g.:
#     DATASET=Beauty bash clean_tiger/run_clean_tiger.sh all
# =============================================================================

# --- repo root (this file lives in <root>/clean_tiger/) ----------------------
CLEAN_TIGER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${CLEAN_TIGER_DIR}/.." && pwd)"

# --- dataset & seeds --------------------------------------------------------
DATASET="${DATASET:-Toys}"          # Toys | Beauty | Sport | Electronics | ...
SEED_RQVAE="${SEED_RQVAE:-2024}"    # 2024 = original TIGER / legacy default
SEED_T5="${SEED_T5:-2025}"

# --- dataset root (interaction parquets: train/valid/test) -------------------
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"

# --- 1. item content representation -----------------------------------------
# Content-ONLY item embedding produced by the Sentence-T5 encoder.
#   * data/<D>/item_emb.parquet          <- THIS is the original TIGER input
#   * data/<D>/fused_item_emb*.parquet   <- TA-SID input (content + behavior)
# The guard in assert_clean_emb() refuses to run on a fused_* file.
CONTENT_EMB="${CONTENT_EMB:-${DATA_ROOT}/${DATASET}/item_emb.parquet}"

# --- 2. RQ-VAE (original TIGER hyper-parameters) -----------------------------
RQVAE_LR="${RQVAE_LR:-1e-3}"
RQVAE_EPOCHS="${RQVAE_EPOCHS:-3000}"
RQVAE_BATCH="${RQVAE_BATCH:-1024}"
RQVAE_NUM_WORKERS="${RQVAE_NUM_WORKERS:-4}"
RQVAE_EVAL_STEP="${RQVAE_EVAL_STEP:-50}"
RQVAE_LEARNER="${RQVAE_LEARNER:-AdamW}"
RQVAE_LR_SCHEDULER="${RQVAE_LR_SCHEDULER:-linear}"
RQVAE_WARMUP_EPOCHS="${RQVAE_WARMUP_EPOCHS:-50}"
RQVAE_WEIGHT_DECAY="${RQVAE_WEIGHT_DECAY:-1e-4}"
RQVAE_DROPOUT="${RQVAE_DROPOUT:-0.0}"
RQVAE_BN="${RQVAE_BN:-False}"
RQVAE_LOSS_TYPE="${RQVAE_LOSS_TYPE:-mse}"
RQVAE_KMEANS_INIT="${RQVAE_KMEANS_INIT:-True}"
RQVAE_KMEANS_ITERS="${RQVAE_KMEANS_ITERS:-100}"
RQVAE_SK_ITERS="${RQVAE_SK_ITERS:-50}"
RQVAE_BETA="${RQVAE_BETA:-0.25}"
RQVAE_QUANT_LOSS_WEIGHT="${RQVAE_QUANT_LOSS_WEIGHT:-1.0}"
RQVAE_SAVE_LIMIT="${RQVAE_SAVE_LIMIT:-5}"
RQVAE_NUM_EMB_LIST=(256 256 256)
RQVAE_E_DIM=32
RQVAE_LAYERS=(512 256 128 64)
RQVAE_SK_EPS=(0.0 0.0 0.003)

# --- 3. historical mechanisms: PINNED OFF -----------------------------------
# Passed explicitly to rqvae/main.py by rqvae_clean_args() below.
#   L_trans                  Track B transition-aware contrastive loss
#   L_balance                Track B codebook balance loss
#   L_align                  E50 b2 InfoNCE alignment to a target embedding
#   --trans_graph_path       transition co-occurrence graph (Track B input)
#   --align_data_path        behavior embedding used as the alignment target
# (behavior embedding / PPMI / SVD / content+behavior fusion enter the pipeline
#  only through the two input files above, so pinning these to 0 plus feeding a
#  content-only embedding is sufficient to keep them out entirely.)
LAMBDA_TRANS=0.0
LAMBDA_BALANCE=0.0
ALIGN_WEIGHT=0.0

# --- 4. Semantic ID generation ----------------------------------------------
SID_OUT="${SID_OUT:-${DATA_ROOT}/${DATASET}/${DATASET}_clean_tiger_sid.npy}"
SID_PREFIX="<a_{}> <b_{}> <c_{}> <d_{}> <e_{}>"   # original TIGER SID format

# --- 5. T5 recommender (original TIGER hyper-parameters) ---------------------
T5_BATCH="${T5_BATCH:-256}"
T5_INFER_SIZE="${T5_INFER_SIZE:-96}"
T5_NUM_EPOCHS="${T5_NUM_EPOCHS:-200}"
T5_LR="${T5_LR:-1e-4}"
T5_NUM_LAYERS="${T5_NUM_LAYERS:-4}"
T5_NUM_DECODER_LAYERS="${T5_NUM_DECODER_LAYERS:-4}"
T5_D_MODEL="${T5_D_MODEL:-128}"
T5_D_FF="${T5_D_FF:-1024}"
T5_NUM_HEADS="${T5_NUM_HEADS:-6}"
T5_D_KV="${T5_D_KV:-64}"
T5_DROPOUT="${T5_DROPOUT:-0.1}"
T5_MAX_LEN="${T5_MAX_LEN:-20}"
T5_BEAM="${T5_BEAM:-30}"
T5_EARLY_STOP="${T5_EARLY_STOP:-10}"
# NOTE: model/main.py declares --topk_list with `type=list`, which argparse
# turns into a list of CHARACTERS for a "5,10,20" string. It is therefore never
# passed; the script's own default [5, 10, 20] (which contains NDCG@20, required
# by its early-stopping branch) is used instead.

# --- 6. output locations (isolated from historical experiments) --------------
CLEAN_CKPT_DIR="${CLEAN_CKPT_DIR:-${CLEAN_TIGER_DIR}/ckpt/${DATASET}}"
CLEAN_LOG_DIR="${CLEAN_LOG_DIR:-${CLEAN_TIGER_DIR}/logs}"
RQVAE_CKPT_DIR="${RQVAE_CKPT_DIR:-${CLEAN_CKPT_DIR}/rqvae}"
T5_CKPT="${T5_CKPT:-${CLEAN_CKPT_DIR}/tiger_clean_${DATASET}_seed${SEED_T5}.pth}"
T5_LOG="${T5_LOG:-${CLEAN_LOG_DIR}/tiger_clean_${DATASET}_seed${SEED_T5}.log}"

# --- helpers -----------------------------------------------------------------
# Explicit clean argument list for rqvae/main.py — the zeros are written out on
# purpose so the invocation is self-documenting in the logs.
rqvae_clean_args() {
  echo "--lambda_trans ${LAMBDA_TRANS}" \
       "--lambda_balance ${LAMBDA_BALANCE}" \
       "--align_weight ${ALIGN_WEIGHT}"
}

# Refuse to run on a content+behavior fused embedding.
assert_clean_emb() {
  local emb="$1"
  if [ ! -f "$emb" ]; then
    echo "✗ [clean-tiger] content embedding not found: $emb" >&2
    return 1
  fi
  case "$(basename "$emb")" in
    fused_item_emb*)
      echo "✗ [clean-tiger] '$emb' is a content+BEHAVIOR fused embedding." >&2
      echo "  Clean TIGER must use the content-only item_emb.parquet." >&2
      return 1 ;;
  esac
  return 0
}

# Newest best_collision_model.pth under an RQ-VAE ckpt root.
newest_rqvae_ckpt() {
  find "$1" -name best_collision_model.pth -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}
