#!/bin/bash
# Fine-tune UniMiSS+ across the 5 grouped-by-patient stratified CV folds for the
# Vietnamese X-ray Normal/Abnormal task, then run a held-out test evaluation per fold.
#
# Model selection uses val.txt only ("test" list during training). test.txt is
# touched exactly once per fold, in the eval_only pass, for final reporting.
set -euo pipefail
cd "$(dirname "$0")"

CV_DIR="${CV_DIR:-../Script/labels/vietnam_xray_cv}"
DATA_ROOT="../Script/UniMiSSPlus_data"
# Optional loss override. LOSS=asl with ASL_GAMMA_POS>0 = focal-style loss.
LOSS="${LOSS:-ce}"
ASL_GAMMA_NEG="${ASL_GAMMA_NEG:-4}"
ASL_GAMMA_POS="${ASL_GAMMA_POS:-2}"
LOSS_FLAGS=(--loss "$LOSS")
[ "$LOSS" = "asl" ] && LOSS_FLAGS+=(--asl_gamma_neg "$ASL_GAMMA_NEG" --asl_gamma_pos "$ASL_GAMMA_POS")
PRETRAIN="../UniMissPlus.pth"
EPOCHS="${EPOCHS:-30}"
# Tuned for 16GB VRAM / 16 threads: at batch 32 the GPU was already at 100%
# util but only 5.6/16GB VRAM used (compute-bound, not memory-bound) -- batch
# 64 amortizes per-step overhead better without starving gradient updates on
# folds with ~760 train images. num_workers 8 uses half the 16 threads.
# Override BATCH_SIZE=32 to reproduce the original seed42/123/2024 numbers
# exactly (they were all run at batch 32).
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
NO_PRETRAIN="${NO_PRETRAIN:-0}"
# seed 42 is the original run (Script/results/unimiss_vietnam_xray); other
# seeds get their own directory so repeated runs don't clobber each other.
# NO_PRETRAIN=1 runs the same recipe from random init, as a negative control.
if [ "$NO_PRETRAIN" = "1" ]; then
  OUT_ROOT="${OUT_ROOT:-../Script/results/unimiss_vietnam_xray_scratch_seed${SEED}}"
  PRETRAIN_FLAGS=()
elif [ "$SEED" = "42" ]; then
  OUT_ROOT="${OUT_ROOT:-../Script/results/unimiss_vietnam_xray}"
  PRETRAIN_FLAGS=(--pre_train --pre_train_path "$PRETRAIN")
else
  OUT_ROOT="${OUT_ROOT:-../Script/results/unimiss_vietnam_xray_seed${SEED}}"
  PRETRAIN_FLAGS=(--pre_train --pre_train_path "$PRETRAIN")
fi

for fold in 0 1 2 3 4; do
  FOLD_DIR="$CV_DIR/fold_${fold}"
  TRAIN_OUT="$OUT_ROOT/fold_${fold}"
  EVAL_OUT="$OUT_ROOT/fold_${fold}_eval"

  echo "=== Fold ${fold}: training (val.txt for model selection) ==="
  python Downstream/2D/Cls/main_flexible.py \
    --task covid --covid_mode selected --covid_classes Abnormal,Normal \
    --covid_root "$DATA_ROOT" \
    --covid_train_list "$FOLD_DIR/train_oversampled.txt" \
    --covid_test_list "$FOLD_DIR/val.txt" \
    --output_dir "$TRAIN_OUT" \
    "${PRETRAIN_FLAGS[@]}" "${LOSS_FLAGS[@]}" \
    --batch_size "$BATCH_SIZE" --test_batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" --epochs "$EPOCHS" --learning_rate 0.0001 \
    --seed "$SEED"

  echo "=== Fold ${fold}: held-out test evaluation (test.txt, touched once) ==="
  python Downstream/2D/Cls/main_flexible.py \
    --task covid --covid_mode selected --covid_classes Abnormal,Normal \
    --covid_root "$DATA_ROOT" \
    --covid_train_list "$FOLD_DIR/train_oversampled.txt" \
    --covid_test_list "$FOLD_DIR/test.txt" \
    --output_dir "$EVAL_OUT" \
    --checkpoint_path "$TRAIN_OUT/best.pth" \
    --eval_only --grad_cam --grad_cam_per_class
done

echo "All folds complete. Results under $OUT_ROOT"
