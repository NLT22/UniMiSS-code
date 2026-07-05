"""Fold-level metrics for the UniMiSS+ fine-tuning CV run, in the same shape as
eval_pretrained_features.py's feature_eval_metrics.csv / feature_eval_predictions.csv,
so both can be combined into one master comparison table.

Usage:
    python summarize_unimiss_cv.py \
        --results-root results/unimiss_vietnam_xray \
        --folds 5 \
        --output-dir results/unimiss_vietnam_xray/summary
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="unimiss_plus_finetuned")
    args = parser.parse_args()

    root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    prediction_rows = []

    for fold_idx in range(args.folds):
        eval_dir = root / f"fold_{fold_idx}_eval"
        npz_path = eval_dir / "eval_predictions.npz"
        if not npz_path.is_file():
            print(f"Missing {npz_path}, skipping fold {fold_idx}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        y_true_abnormal_idx = data["label_names"].tolist().index("Abnormal")
        y_true = (data["y_true"] == y_true_abnormal_idx).astype(int)  # 1 = Abnormal
        y_score = data["y_score"][:, y_true_abnormal_idx]
        y_pred = (y_score >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        abnormal_recall = recall_score(y_true, y_pred, pos_label=1)
        normal_specificity = recall_score(y_true, y_pred, pos_label=0)

        metrics_rows.append({
            "method": args.method_name, "fold": str(fold_idx),
            "test_auc": auc, "test_ap": ap,
            "test_balanced_accuracy": balanced_acc,
            "test_abnormal_recall": abnormal_recall,
            "test_normal_specificity": normal_specificity,
            "n_test": len(y_true),
        })

        best_metrics_path = root / f"fold_{fold_idx}" / "best_metrics.json"
        val_info = json.loads(best_metrics_path.read_text()) if best_metrics_path.is_file() else {}
        metrics_rows[-1]["val_macro_auc_at_best_epoch"] = val_info.get("macro_auc")
        metrics_rows[-1]["best_epoch"] = val_info.get("epoch")

        for yt, ys, yp in zip(y_true, y_score, y_pred):
            prediction_rows.append({
                "method": args.method_name, "fold": str(fold_idx),
                "y_true_abnormal": int(yt), "y_score_abnormal": float(ys), "y_pred_abnormal": int(yp),
            })

    if not metrics_rows:
        raise SystemExit("No fold eval results found. Has the fine-tuning run finished?")

    metrics_path = output_dir / "unimiss_cv_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    predictions_path = output_dir / "unimiss_cv_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    aucs = [r["test_auc"] for r in metrics_rows]
    aps = [r["test_ap"] for r in metrics_rows]
    print(f"UniMiSS+ pooled-fold AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"UniMiSS+ pooled-fold AP:  {np.mean(aps):.3f} +/- {np.std(aps):.3f}")
    print(f"Wrote {metrics_path} and {predictions_path}")


if __name__ == "__main__":
    main()
