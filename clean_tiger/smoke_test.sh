#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — end-to-end smoke test of the frozen Clean TIGER baseline
# =============================================================================
# Runs all four real stages (RQ-VAE -> Semantic ID -> T5 -> evaluation) with
# tiny budgets, on a truncated interaction set. It exercises the true code
# paths; it is not a substitute for a full training run.
#
# The item-content embedding is the real, full one (Symlink into the smoke
# dir), so the RQ-VAE and the Semantic ID stage are genuinely exercised; only
# the interaction parquets are truncated so that the T5 stage + beam-search
# evaluation finish in seconds.
#
# Usage:
#   bash clean_tiger/smoke_test.sh            # default DATASET=Toys
#   DATASET=Beauty bash clean_tiger/smoke_test.sh
#
# Artefacts land in clean_tiger/_smoke/ and are safe to delete.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"

DATASET="${DATASET:-Toys}"
USERS="${SMOKE_USERS:-60}"
SMOKE_ROOT="${HERE}/_smoke/${DATASET}"
SMOKE_DATA="${SMOKE_ROOT}/data"          # mirrors the real data/ layout
SMOKE_DS="${SMOKE_DATA}/${DATASET}"      # .../data/<DATASET>/

log() { echo "[$(date '+%F %T')] [clean-tiger smoke] $*"; }

log "preparing truncated interaction set (${USERS} users) in ${SMOKE_DS}"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_DS"

DATASET="$DATASET" USERS="$USERS" SRC_DIR="${ROOT}/data/${DATASET}" DST_DIR="$SMOKE_DS" \
python -I - <<'PY'
import collections, os, pandas as pd
src, dst, n = os.environ["SRC_DIR"], os.environ["DST_DIR"], int(os.environ["USERS"])

# Two deliberate simplifications, both needed only to make the *plumbing* test
# finish in seconds. This is NOT a generalisation measurement:
#
#  (1) all three splits are the same rows, and
#  (2) items are projected onto the 5 most frequent item IDs (a deterministic
#      map, so it is reproducible across runs).
#
# model/main.py writes a checkpoint only when validation NDCG@20 improves on the
# running best (0.0). NDCG here is an exact 4-token Semantic-ID match, so on the
# full item vocabulary a barely-trained tiny model scores exactly 0 forever and
# never saves — which would leave the evaluation stage with nothing to run.
# Shrinking the target vocabulary makes the task learnable within a few epochs,
# so the train -> checkpoint -> evaluation path is genuinely exercised.
head = pd.read_parquet(f"{src}/train.parquet").head(n)
counts = collections.Counter()
for row in head.itertuples(index=False):
    counts.update(list(row.history) + [row.target])
top = [item for item, _ in counts.most_common(5)]
rank = {item: k for k, item in enumerate(top)}
pick = lambda x: top[rank[x]] if x in rank else top[int(x) % len(top)]
print(f"  projecting onto item vocabulary: {top}")
head = head.assign(
    history=head["history"].apply(lambda s: [pick(x) for x in s]),
    target=head["target"].apply(pick),
)
for split in ("train", "valid", "test"):
    head.to_parquet(f"{dst}/{split}.parquet", index=False)
    print(f"  {split}: {len(head)} rows")
PY

# real, full content-only embedding — symlinked, not copied
ln -sf "${ROOT}/data/${DATASET}/item_emb.parquet" "${SMOKE_DS}/item_emb.parquet"

# ---- tiny budgets + isolated outputs ---------------------------------------
export DATA_ROOT="$SMOKE_DATA"
export RQVAE_EPOCHS=2
export RQVAE_EVAL_STEP=1
export RQVAE_SAVE_LIMIT=1
export RQVAE_BATCH=1024
export RQVAE_NUM_WORKERS=2
export SID_OUT="${SMOKE_ROOT}/${DATASET}_clean_tiger_sid.npy"
# model/main.py only writes a checkpoint when validation NDCG@20 improves on the
# running best (0.0 initially). On this tiny set a few epochs are enough to
# overfit a non-zero hit, which is what lets the eval stage run for real.
export T5_NUM_EPOCHS=60
export T5_BATCH=32
export T5_INFER_SIZE=32
export T5_BEAM=10
export T5_EARLY_STOP=60
export T5_CKPT="${SMOKE_ROOT}/tiger_clean_smoke.pth"
export T5_LOG="${SMOKE_ROOT}/tiger_clean_smoke.log"
export CLEAN_LOG_DIR="${SMOKE_ROOT}/logs"
mkdir -p "$CLEAN_LOG_DIR"

log "running Clean TIGER stages: rqvae -> sid -> t5 -> eval"
bash "${HERE}/run_clean_tiger.sh" all

log "running contamination verification"
DATASET="$DATASET" bash "${HERE}/run_clean_tiger.sh" verify

# ---- assert the artefacts actually exist -----------------------------------
fail=0
for f in "$SID_OUT" "$T5_CKPT"; do
  if [ -s "$f" ]; then log "OK   ${f}"; else log "MISSING ${f}"; fail=1; fi
done
[ -s "${CLEAN_LOG_DIR}/t5_${DATASET}_seed${SEED_T5}_eval.log" ] \
  && log "OK   evaluation log written" \
  || { log "MISSING evaluation log"; fail=1; }

echo "=================================================================="
if [ "$fail" -eq 0 ]; then
  echo "[clean-tiger smoke] PASS"
else
  echo "[clean-tiger smoke] FAIL"
fi
echo "=================================================================="
exit "$fail"
