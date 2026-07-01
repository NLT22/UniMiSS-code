#!/usr/bin/env python3
"""
Summarize Vietnamese X-ray CV evaluation outputs.

Inputs are the CSV files produced by eval_pretrained_features.py. Outputs are a
compact report table, pooled out-of-fold predictions, optional ROC/PR plots, and
pooled confusion matrices.
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


CLASS_ABNORMAL = 0
CLASS_NORMAL = 1

METHOD_META = {
    "torchxrayvision_densenet121": {
        "citation": "Cohen2022_TorchXRayVision",
        "pretraining_source": "Chest X-ray datasets through TorchXRayVision",
        "adaptation": "Frozen DenseNet121 embeddings + logistic regression",
    },
    "imagenet_densenet121": {
        "citation": "Russakovsky2015_ImageNet; Huang2017_DenseNet",
        "pretraining_source": "ImageNet ILSVRC",
        "adaptation": "Frozen DenseNet121 embeddings + logistic regression",
    },
    "imagenet_resnet50": {
        "citation": "Russakovsky2015_ImageNet; He2016_ResNet",
        "pretraining_source": "ImageNet ILSVRC",
        "adaptation": "Frozen ResNet50 embeddings + logistic regression",
    },
    "rad_dino": {
        "citation": "PerezGarcia2025_RAD_DINO",
        "pretraining_source": "Self-supervised chest X-ray / biomedical image pretraining",
        "adaptation": "Frozen RAD-DINO CLS embeddings + logistic regression",
    },
    "medsiglip": {
        "citation": "Google_MedSigLIP_ModelCard",
        "pretraining_source": "MedSigLIP medical image-text pretraining",
        "adaptation": "Frozen image embeddings + logistic regression",
    },
    "unimissplus_finetune": {
        "citation": "Xie2024_UniMiSSPlus; Xie2022_UniMiSS",
        "pretraining_source": "UniMiSS+ / UniMiSS pretrained weights",
        "adaptation": "Fine-tuning with validation-controlled model selection",
    },
}

SUMMARY_METRICS = [
    ("auc", "AUC"),
    ("average_precision", "AP"),
    ("balanced_accuracy", "Balanced Acc"),
    ("abnormal_recall", "Abnormal Recall"),
    ("specificity", "Specificity"),
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def mean_sd(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return math.nan, math.nan
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def fmt(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.3f}"


def fmt_mean_sd(mean: float, sd: float) -> str:
    if math.isnan(mean):
        return "NA"
    return f"{mean:.3f}+/-{sd:.3f}"


def combine_metrics(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for row in read_csv(path):
            row["source_file"] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError("No metric rows found")
    return rows


def combine_predictions(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.exists():
            for row in read_csv(path):
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def summarize_metrics(metric_rows: list[dict]) -> list[dict]:
    by_method = defaultdict(list)
    for row in metric_rows:
        by_method[row["method"]].append(row)

    summary_rows = []
    for method, rows in sorted(by_method.items()):
        meta = METHOD_META.get(method, {})
        summary = {
            "method": method,
            "citation": rows[0].get("citation_key") or meta.get("citation", ""),
            "pretraining_source": meta.get("pretraining_source", ""),
            "adaptation": meta.get("adaptation", ""),
            "folds": str(len(rows)),
        }
        for metric, _ in SUMMARY_METRICS:
            mean, sd = mean_sd([parse_float(row.get(metric, "")) for row in rows])
            summary[f"{metric}_mean"] = "" if math.isnan(mean) else f"{mean:.6f}"
            summary[f"{metric}_sd"] = "" if math.isnan(sd) else f"{sd:.6f}"
            summary[f"{metric}_mean_sd"] = fmt_mean_sd(mean, sd)
        summary_rows.append(summary)
    return summary_rows


def write_markdown(path: Path, summary_rows: list[dict]):
    headers = [
        "Method",
        "Citation",
        "Pretraining source",
        "Adaptation",
        "AUC mean+/-SD",
        "AP mean+/-SD",
        "Balanced Acc mean+/-SD",
        "Abnormal Recall mean+/-SD",
        "Specificity mean+/-SD",
    ]
    lines = [
        "# Cross-Validation Summary",
        "",
        "Abnormal is the positive class for AUC, average precision, recall, and F1.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        values = [
            row["method"],
            row["citation"],
            row["pretraining_source"],
            row["adaptation"],
            row["auc_mean_sd"],
            row["average_precision_mean_sd"],
            row["balanced_accuracy_mean_sd"],
            row["abnormal_recall_mean_sd"],
            row["specificity_mean_sd"],
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    lines.extend([
        "",
        "Accuracy is included in `cv_summary.csv` but is not the primary metric because the dataset is imbalanced.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pooled_confusion_rows(prediction_rows: list[dict]) -> list[dict]:
    by_method = defaultdict(list)
    for row in prediction_rows:
        by_method[row["method"]].append(row)

    rows = []
    for method, method_rows in sorted(by_method.items()):
        counts = defaultdict(int)
        for row in method_rows:
            true_label = int(row["true_label"])
            pred_label = int(row["pred_label"])
            counts[(true_label, pred_label)] += 1
        abnormal_total = counts[(CLASS_ABNORMAL, CLASS_ABNORMAL)] + counts[(CLASS_ABNORMAL, CLASS_NORMAL)]
        normal_total = counts[(CLASS_NORMAL, CLASS_ABNORMAL)] + counts[(CLASS_NORMAL, CLASS_NORMAL)]
        abnormal_recall = (
            counts[(CLASS_ABNORMAL, CLASS_ABNORMAL)] / abnormal_total if abnormal_total else math.nan
        )
        specificity = counts[(CLASS_NORMAL, CLASS_NORMAL)] / normal_total if normal_total else math.nan
        rows.append({
            "method": method,
            "abnormal_pred_abnormal": str(counts[(CLASS_ABNORMAL, CLASS_ABNORMAL)]),
            "abnormal_pred_normal": str(counts[(CLASS_ABNORMAL, CLASS_NORMAL)]),
            "normal_pred_abnormal": str(counts[(CLASS_NORMAL, CLASS_ABNORMAL)]),
            "normal_pred_normal": str(counts[(CLASS_NORMAL, CLASS_NORMAL)]),
            "pooled_abnormal_recall": fmt(abnormal_recall),
            "pooled_specificity": fmt(specificity),
        })
    return rows


def write_plots(prediction_rows: list[dict], output_dir: Path):
    if not prediction_rows:
        return
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, precision_recall_curve, roc_curve
    except Exception as exc:
        print(f"Skipping ROC/PR plots because plotting dependencies are unavailable: {exc}")
        return

    by_method = defaultdict(list)
    for row in prediction_rows:
        by_method[row["method"]].append(row)

    plt.figure(figsize=(7, 6))
    for method, rows in sorted(by_method.items()):
        y_true = [1 if int(row["true_label"]) == CLASS_ABNORMAL else 0 for row in rows]
        y_score = [float(row["abnormal_score"]) for row in rows]
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{method} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.6", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Pooled Out-of-Fold ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_roc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    for method, rows in sorted(by_method.items()):
        y_true = [1 if int(row["true_label"]) == CLASS_ABNORMAL else 0 for row in rows]
        y_score = [float(row["abnormal_score"]) for row in rows]
        if len(set(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=f"{method}")
    plt.xlabel("Abnormal recall")
    plt.ylabel("Precision")
    plt.title("Pooled Out-of-Fold Precision-Recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pooled_pr.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize CV metrics and pooled predictions")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more feature_eval_metrics.csv files",
    )
    parser.add_argument(
        "--predictions",
        nargs="*",
        default=[],
        help="Optional feature_eval_predictions.csv files",
    )
    parser.add_argument("--output-dir", default="results/cv_summary", help="Output summary directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metric_rows = combine_metrics([Path(path) for path in args.metrics])
    summary_rows = summarize_metrics(metric_rows)

    csv_fields = [
        "method", "citation", "pretraining_source", "adaptation", "folds",
    ]
    for metric, _ in SUMMARY_METRICS:
        csv_fields.extend([f"{metric}_mean", f"{metric}_sd", f"{metric}_mean_sd"])
    write_csv(output_dir / "cv_summary.csv", summary_rows, csv_fields)
    write_markdown(output_dir / "cv_summary.md", summary_rows)

    prediction_rows = combine_predictions([Path(path) for path in args.predictions])
    if prediction_rows:
        prediction_fields = list(prediction_rows[0].keys())
        write_csv(output_dir / "pooled_predictions.csv", prediction_rows, prediction_fields)
        confusion_rows = pooled_confusion_rows(prediction_rows)
        write_csv(
            output_dir / "pooled_confusion_matrices.csv",
            confusion_rows,
            [
                "method", "abnormal_pred_abnormal", "abnormal_pred_normal",
                "normal_pred_abnormal", "normal_pred_normal",
                "pooled_abnormal_recall", "pooled_specificity",
            ],
        )
        write_plots(prediction_rows, output_dir)

    print(f"Saved summary CSV: {output_dir / 'cv_summary.csv'}")
    print(f"Saved summary Markdown: {output_dir / 'cv_summary.md'}")
    if prediction_rows:
        print(f"Saved pooled predictions: {output_dir / 'pooled_predictions.csv'}")
        print(f"Saved pooled confusion matrices: {output_dir / 'pooled_confusion_matrices.csv'}")


if __name__ == "__main__":
    main()
