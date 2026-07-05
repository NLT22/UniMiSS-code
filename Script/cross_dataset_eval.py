"""Cross-dataset zero-shot evaluation: models trained on large public CXR
datasets, evaluated directly on our Vietnamese X-ray set (no fine-tuning on
our data).

Uses downstream checkpoints trained earlier on NIH ChestX-ray14 (15-class
multi-label), COVID-19 Radiography and COVID-QU-Ex (3-class). Each model's
output is reduced to a single "Abnormal" score and scored against our binary
labels (Abnormal = positive). This is a true external test: the models never
saw our data.

Mapping to Abnormal score:
  NIH   -> 1 - sigmoid(logit["No Finding"])          (idx 14)
  COVID -> 1 - softmax(logits)["Normal"]             (idx 2)

Usage:
    python cross_dataset_eval.py --old-results <old_project>/UniMiSSPlus/results \
        --data-root Script/UniMiSSPlus_data \
        --manifest Script/labels/vietnam_xray_cv/cv_manifest.csv \
        --output Script/results/cross_dataset_eval.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "UniMiSSPlus/Downstream/2D/Cls"))
from net.MiTPlus_encoder import MiTPlus_encoder  # noqa: E402
from dataset.flexible_datasets import build_transform_classification  # noqa: E402

# (checkpoint subdir, num_classes, reduce-fn name, index)
MODELS = {
    "NIH-ChestXray14": ("unimiss_nih/best.pth", 15, "sigmoid_nofinding", 14),
    "COVID19-Radiography": ("covid_other_normal/best.pth", 3, "softmax_normal", 2),
    "COVID-QU-Ex": ("train_quex/best.pth", 3, "softmax_normal", 2),
}


def load_labels(manifest_path):
    """image_path -> y_abnormal (1 if Abnormal else 0). Uses test rows (each sample once)."""
    labels = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            # normal_abnormal_class: 0 = Abnormal, 1 = Normal -> positive=Abnormal
            labels[r["image_path"]] = 1 if r["normal_abnormal_class"] == "0" else 0
    return labels


@torch.no_grad()
def eval_model(name, ckpt_rel, num_classes, reduce_name, idx, old_results, data_root, labels, device):
    model = MiTPlus_encoder(num_classes=num_classes)
    ck = torch.load(Path(old_results) / ckpt_rel, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    tf = build_transform_classification(normalize="chestx-ray", crop_size=224, resize=256,
                                        mode="test", test_augment=False)

    paths = sorted(labels.keys())
    y_true, y_score = [], []
    for i in range(0, len(paths), 32):
        batch = paths[i:i + 32]
        imgs = torch.stack([tf(Image.open(data_root / p).convert("RGB")) for p in batch]).to(device)
        out = model(imgs)
        if reduce_name == "sigmoid_nofinding":
            score = 1.0 - torch.sigmoid(out)[:, idx]           # 1 - P(No Finding)
        else:
            score = 1.0 - torch.softmax(out, dim=1)[:, idx]    # 1 - P(Normal)
        y_score.extend(score.cpu().numpy().tolist())
        y_true.extend(labels[p] for p in batch)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "model": name,
        "trained_on": name,
        "n": len(y_true),
        "auc": round(roc_auc_score(y_true, y_score), 4),
        "ap": round(average_precision_score(y_true, y_score), 4),
        "balanced_acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "abnormal_recall": round(recall_score(y_true, y_pred, pos_label=1), 4),
        "normal_specificity": round(recall_score(y_true, y_pred, pos_label=0), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-results", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    labels = load_labels(args.manifest)
    data_root = Path(args.data_root)
    print(f"Evaluating on {len(labels)} Vietnamese X-ray images (Abnormal = positive).")

    rows = []
    for name, (ckpt, nc, red, idx) in MODELS.items():
        try:
            r = eval_model(name, ckpt, nc, red, idx, args.old_results, data_root, labels, args.device)
            rows.append(r)
            print(f"  {name:22s} AUC={r['auc']:.3f} AP={r['ap']:.3f} "
                  f"balAcc={r['balanced_acc']:.3f} AbnRecall={r['abnormal_recall']:.3f} "
                  f"NorSpec={r['normal_specificity']:.3f}")
        except Exception as e:
            print(f"  {name}: FAILED {e!r}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
