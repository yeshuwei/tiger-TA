#!/bin/bash
# TA-SID Pipeline Launcher — runs Sport then Toys in sequence
cd "$(dirname "$0")/.." || exit 1

echo "============================================"
echo "TA-SID Pipeline started at: $(date)"
echo "============================================"

echo ""
echo "========== [1/2] Running SPORT TA-SID =========="
python rqvae/run_ta_sid_pipeline.py --dataset Sport 2>&1
SPORT_EXIT=$?
echo "Sport pipeline exit code: $SPORT_EXIT"

if [ $SPORT_EXIT -ne 0 ]; then
    echo "⚠ Sport failed (code $SPORT_EXIT), continuing with Toys anyway..."
fi

echo ""
echo "========== [2/2] Running TOYS TA-SID =========="
python rqvae/run_ta_sid_pipeline.py --dataset Toys 2>&1
TOYS_EXIT=$?
echo "Toys pipeline exit code: $TOYS_EXIT"

echo ""
echo "============================================"
echo "TA-SID Pipeline completed at: $(date)"
echo "Sport exit: $SPORT_EXIT, Toys exit: $TOYS_EXIT"
echo "============================================"
