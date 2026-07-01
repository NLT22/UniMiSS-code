#!/usr/bin/env python3
"""
Evaluate pretrained image embeddings on Vietnamese X-ray Normal/Abnormal CV folds.

This script intentionally does not train a CNN from random initialization. It
extracts frozen features from public pretrained models and trains only a
logistic-regression classifier on each train fold.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


CLASS_ABNORMAL = 0
CLASS_NORMAL = 1

METHOD_CITATIONS = {
    "torchxrayvision_densenet121": "Cohen2022_TorchXRayVision",
    "imagenet_densenet121": "Russakovsky2015_ImageNet; Huang2017_DenseNet",
    "imagenet_resnet50": "Russakovsky2015_ImageNet; He2016_ResNet",
    "rad_dino": "PerezGarcia2025_RAD_DINO",
    "medsiglip": "Google_MedSigLIP_ModelCard",
}


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    valid = set(METHOD_CITATIONS)
    invalid = [method for method in methods if method not in valid]
    if invalid:
        raise ValueError(f"Unknown method(s): {', '.join(invalid)}. Valid: {', '.join(sorted(valid))}")
    return methods


def parse_c_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid:
        raise ValueError("--c-grid must contain at least one value")
    return grid


def parse_labeled_list(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                image_path, label = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Invalid list line in {path}:{line_no}: {line}") from exc
            label_int = int(label)
            if label_int not in (CLASS_ABNORMAL, CLASS_NORMAL):
                raise ValueError(f"Unsupported label {label_int} in {path}:{line_no}; expected 0 or 1")
            rows.append({
                "image_path": image_path,
                "label": label_int,
            })
    return rows


def fold_number(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return 999999


def load_cv_folds(cv_dir: Path) -> list[dict]:
    folds = []
    for fold_dir in sorted(cv_dir.glob("fold_*"), key=fold_number):
        if not fold_dir.is_dir():
            continue
        required = {
            "train": fold_dir / "train.txt",
            "train_oversampled": fold_dir / "train_oversampled.txt",
            "val": fold_dir / "val.txt",
            "test": fold_dir / "test.txt",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CV list files in {fold_dir}: {', '.join(missing)}")
        folds.append({
            "fold": fold_number(fold_dir),
            "dir": fold_dir,
            "train": parse_labeled_list(required["train"]),
            "train_oversampled": parse_labeled_list(required["train_oversampled"]),
            "val": parse_labeled_list(required["val"]),
            "test": parse_labeled_list(required["test"]),
        })
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found under {cv_dir}")
    return folds


def resolved_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return data_root / image_path


def unique_image_paths(folds: list[dict]) -> list[str]:
    seen = set()
    ordered = []
    for fold in folds:
        for split in ("train", "train_oversampled", "val", "test"):
            for row in fold[split]:
                image_path = row["image_path"]
                if image_path not in seen:
                    seen.add(image_path)
                    ordered.append(image_path)
    return ordered


def path_digest(paths: list[str], method: str) -> str:
    digest = hashlib.sha256()
    digest.update(method.encode("utf-8"))
    for path in paths:
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
    return digest.hexdigest()[:16]


def load_feature_cache(cache_path: Path) -> dict[str, list[float]] | None:
    if not cache_path.exists():
        return None
    import numpy as np

    payload = np.load(cache_path, allow_pickle=False)
    paths = payload["paths"].tolist()
    features = payload["features"]
    return {path: features[idx] for idx, path in enumerate(paths)}


def save_feature_cache(cache_path: Path, paths: list[str], features_by_path: dict):
    import numpy as np

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix = np.vstack([features_by_path[path] for path in paths]).astype("float32")
    np.savez_compressed(cache_path, paths=np.array(paths), features=feature_matrix)


def make_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def pil_loader_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def pil_loader_xray(path: Path):
    from PIL import Image
    import numpy as np
    import torch

    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((224, 224), Image.BILINEAR)
        array = np.asarray(image, dtype="float32")
    array = array / 255.0 * 2048.0 - 1024.0
    return torch.from_numpy(array[None, :, :])


def imagenet_feature_components(method: str, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if method == "imagenet_densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        weights = DenseNet121_Weights.IMAGENET1K_V1
        base = densenet121(weights=weights)
        transform = weights.transforms()

        class DenseNetFeature(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = model.features

            def forward(self, x):
                x = self.features(x)
                x = F.relu(x, inplace=False)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                return torch.flatten(x, 1)

        model = DenseNetFeature(base)
    elif method == "imagenet_resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        base = resnet50(weights=weights)
        transform = weights.transforms()

        class ResNetFeature(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = nn.Sequential(*list(model.children())[:-1])

            def forward(self, x):
                x = self.features(x)
                return torch.flatten(x, 1)

        model = ResNetFeature(base)
    else:
        raise ValueError(f"Unsupported ImageNet method: {method}")

    model.eval().to(device)
    return model, transform


def torchxrayvision_feature_components(weights_name: str, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchxrayvision as xrv

    base = xrv.models.DenseNet(weights=weights_name)

    class XRVFeature(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            if hasattr(self.model, "features"):
                x = self.model.features(x)
                if x.ndim == 4:
                    x = F.relu(x, inplace=False)
                    x = F.adaptive_avg_pool2d(x, (1, 1))
                    return torch.flatten(x, 1)
                return x
            output = self.model(x)
            if isinstance(output, dict):
                for key in ("feats", "features", "embedding", "out"):
                    if key in output:
                        return output[key]
            return output

    model = XRVFeature(base)
    model.eval().to(device)
    return model


def medsiglip_feature_components(model_id: str, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval().to(device)
    return model, processor


def rad_dino_feature_components(model_id: str, device):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval().to(device)
    return model, processor


def batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def extract_imagenet_features(method: str, paths: list[str], data_root: Path, device, batch_size: int) -> dict:
    import numpy as np
    import torch

    model, transform = imagenet_feature_components(method, device)
    features_by_path = {}
    with torch.no_grad():
        for batch_paths in batches(paths, batch_size):
            images = [transform(pil_loader_rgb(resolved_path(data_root, path))) for path in batch_paths]
            tensor = torch.stack(images).to(device)
            features = model(tensor).detach().cpu().numpy().astype("float32")
            for image_path, feature in zip(batch_paths, features):
                features_by_path[image_path] = feature
    if not features_by_path:
        return {}
    first_dim = len(next(iter(features_by_path.values())))
    print(f"  Extracted {len(features_by_path)} {method} feature vectors, dim={first_dim}")
    return features_by_path


def extract_torchxrayvision_features(
    paths: list[str],
    data_root: Path,
    device,
    batch_size: int,
    weights_name: str,
) -> dict:
    import torch

    model = torchxrayvision_feature_components(weights_name, device)
    features_by_path = {}
    with torch.no_grad():
        for batch_paths in batches(paths, batch_size):
            images = [pil_loader_xray(resolved_path(data_root, path)) for path in batch_paths]
            tensor = torch.stack(images).to(device)
            features = model(tensor).detach().cpu().numpy().astype("float32")
            for image_path, feature in zip(batch_paths, features):
                features_by_path[image_path] = feature
    if not features_by_path:
        return {}
    first_dim = len(next(iter(features_by_path.values())))
    print(f"  Extracted {len(features_by_path)} torchxrayvision feature vectors, dim={first_dim}")
    return features_by_path


def extract_medsiglip_features(
    paths: list[str],
    data_root: Path,
    device,
    batch_size: int,
    model_id: str,
) -> dict:
    import torch

    model, processor = medsiglip_feature_components(model_id, device)
    features_by_path = {}
    with torch.no_grad():
        for batch_paths in batches(paths, batch_size):
            images = [pil_loader_rgb(resolved_path(data_root, path)) for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if hasattr(model, "get_image_features"):
                features = model.get_image_features(**inputs)
            else:
                output = model(**inputs)
                if hasattr(output, "image_embeds"):
                    features = output.image_embeds
                elif hasattr(output, "pooler_output"):
                    features = output.pooler_output
                else:
                    raise RuntimeError("Could not find image embeddings in MedSigLIP model output")
            features = features.detach().cpu().numpy().astype("float32")
            for image_path, feature in zip(batch_paths, features):
                features_by_path[image_path] = feature
    if not features_by_path:
        return {}
    first_dim = len(next(iter(features_by_path.values())))
    print(f"  Extracted {len(features_by_path)} MedSigLIP feature vectors, dim={first_dim}")
    return features_by_path


def extract_rad_dino_features(
    paths: list[str],
    data_root: Path,
    device,
    batch_size: int,
    model_id: str,
) -> dict:
    import torch

    model, processor = rad_dino_feature_components(model_id, device)
    features_by_path = {}
    with torch.no_grad():
        for batch_paths in batches(paths, batch_size):
            images = [pil_loader_rgb(resolved_path(data_root, path)) for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                features = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                features = output.last_hidden_state[:, 0, :]
            else:
                raise RuntimeError("Could not find CLS embeddings in RAD-DINO model output")
            features = features.detach().cpu().numpy().astype("float32")
            for image_path, feature in zip(batch_paths, features):
                features_by_path[image_path] = feature
    if not features_by_path:
        return {}
    first_dim = len(next(iter(features_by_path.values())))
    print(f"  Extracted {len(features_by_path)} RAD-DINO feature vectors, dim={first_dim}")
    return features_by_path


def validate_image_paths(paths: list[str], data_root: Path):
    missing = [path for path in paths if not resolved_path(data_root, path).exists()]
    if missing:
        shown = "\n  ".join(missing[:20])
        raise FileNotFoundError(f"{len(missing)} image files from CV lists do not exist under {data_root}:\n  {shown}")


def extract_features(method: str, paths: list[str], data_root: Path, output_dir: Path, args) -> dict:
    cache_path = output_dir / "feature_cache" / f"{method}_{path_digest(paths, method)}.npz"
    if not args.no_cache:
        cached = load_feature_cache(cache_path)
        if cached is not None:
            print(f"  Loaded feature cache: {cache_path}")
            return cached

    device = make_device(args.device)
    print(f"  Extracting {method} on {device}")
    if method.startswith("imagenet_"):
        features = extract_imagenet_features(method, paths, data_root, device, args.batch_size)
    elif method == "torchxrayvision_densenet121":
        features = extract_torchxrayvision_features(paths, data_root, device, args.batch_size, args.xrv_weights)
    elif method == "rad_dino":
        features = extract_rad_dino_features(paths, data_root, device, args.batch_size, args.rad_dino_model)
    elif method == "medsiglip":
        features = extract_medsiglip_features(paths, data_root, device, args.batch_size, args.medsiglip_model)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not args.no_cache:
        save_feature_cache(cache_path, paths, features)
        print(f"  Saved feature cache: {cache_path}")
    return features


def rows_to_arrays(rows: list[dict], features_by_path: dict):
    import numpy as np

    x = np.vstack([features_by_path[row["image_path"]] for row in rows]).astype("float32")
    y = np.array([row["label"] for row in rows], dtype="int64")
    return x, y


def class_count_text(rows: list[dict]) -> str:
    counts = Counter(row["label"] for row in rows)
    return f"Abnormal={counts.get(CLASS_ABNORMAL, 0)}, Normal={counts.get(CLASS_NORMAL, 0)}"


def abnormal_probability(model, x):
    probabilities = model.predict_proba(x)
    abnormal_index = list(model.classes_).index(CLASS_ABNORMAL)
    return probabilities[:, abnormal_index]


def safe_auc(y, score):
    from sklearn.metrics import roc_auc_score

    if len(set(y.tolist())) < 2:
        return math.nan
    return float(roc_auc_score((y == CLASS_ABNORMAL).astype(int), score))


def safe_average_precision(y, score):
    from sklearn.metrics import average_precision_score

    if len(set(y.tolist())) < 2:
        return math.nan
    return float(average_precision_score((y == CLASS_ABNORMAL).astype(int), score))


def train_logistic_regression(x_train, y_train, x_val, y_val, c_grid: list[float], class_weight: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class_weight_arg = None if class_weight == "none" else class_weight
    best = None
    for c_value in c_grid:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight=class_weight_arg,
                max_iter=2000,
                solver="lbfgs",
            ),
        )
        model.fit(x_train, y_train)
        val_score = abnormal_probability(model, x_val)
        auc = safe_auc(y_val, val_score)
        ap = safe_average_precision(y_val, val_score)
        rank = (
            -1.0 if math.isnan(auc) else auc,
            -1.0 if math.isnan(ap) else ap,
            -c_value,
        )
        if best is None or rank > best["rank"]:
            best = {
                "model": model,
                "c": c_value,
                "val_auc": auc,
                "val_average_precision": ap,
                "rank": rank,
            }
    return best


def compute_metrics(y_true, y_pred, score) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[CLASS_ABNORMAL, CLASS_NORMAL])
    abnormal_total = cm[0, 0] + cm[0, 1]
    normal_total = cm[1, 0] + cm[1, 1]
    abnormal_recall = cm[0, 0] / abnormal_total if abnormal_total else math.nan
    specificity = cm[1, 1] / normal_total if normal_total else math.nan
    return {
        "auc": safe_auc(y_true, score),
        "average_precision": safe_average_precision(y_true, score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "abnormal_recall": float(abnormal_recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=CLASS_ABNORMAL, zero_division=0)),
        "cm_abnormal_pred_abnormal": int(cm[0, 0]),
        "cm_abnormal_pred_normal": int(cm[0, 1]),
        "cm_normal_pred_abnormal": int(cm[1, 0]),
        "cm_normal_pred_normal": int(cm[1, 1]),
    }


def value_or_blank(value):
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: value_or_blank(row.get(field, "")) for field in fieldnames})


def evaluate_method(method: str, folds: list[dict], features_by_path: dict, args) -> tuple[list[dict], list[dict]]:
    metrics_rows = []
    prediction_rows = []
    c_grid = parse_c_grid(args.c_grid)

    for fold in folds:
        train_key = "train_oversampled" if args.use_oversampled_train else "train"
        train_rows = fold[train_key]
        val_rows = fold["val"]
        test_rows = fold["test"]

        x_train, y_train = rows_to_arrays(train_rows, features_by_path)
        x_val, y_val = rows_to_arrays(val_rows, features_by_path)
        x_test, y_test = rows_to_arrays(test_rows, features_by_path)

        best = train_logistic_regression(x_train, y_train, x_val, y_val, c_grid, args.class_weight)
        model = best["model"]
        test_score = abnormal_probability(model, x_test)
        test_pred = model.predict(x_test)
        metrics = compute_metrics(y_test, test_pred, test_score)

        row = {
            "method": method,
            "citation_key": METHOD_CITATIONS[method],
            "fold": fold["fold"],
            "best_c": best["c"],
            "val_auc_for_c": best["val_auc"],
            "val_average_precision_for_c": best["val_average_precision"],
            "train_list": train_key,
            "train_samples": len(train_rows),
            "train_unique_images": len({row["image_path"] for row in train_rows}),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "train_counts": class_count_text(train_rows),
            "val_counts": class_count_text(val_rows),
            "test_counts": class_count_text(test_rows),
        }
        row.update(metrics)
        metrics_rows.append(row)

        for source_row, label, pred, score in zip(test_rows, y_test, test_pred, test_score):
            prediction_rows.append({
                "method": method,
                "fold": fold["fold"],
                "image_path": source_row["image_path"],
                "true_label": int(label),
                "true_label_name": "Abnormal" if label == CLASS_ABNORMAL else "Normal",
                "pred_label": int(pred),
                "pred_label_name": "Abnormal" if pred == CLASS_ABNORMAL else "Normal",
                "abnormal_score": float(score),
            })
        print(
            f"  Fold {fold['fold']}: "
            f"AUC={metrics['auc']:.4f} AP={metrics['average_precision']:.4f} "
            f"BalAcc={metrics['balanced_accuracy']:.4f}"
        )

    return metrics_rows, prediction_rows


def write_run_config(path: Path, args, methods: list[str]):
    payload = {
        "data_root": str(Path(args.data_root).resolve()),
        "cv_dir": str(Path(args.cv_dir).resolve()),
        "methods": methods,
        "batch_size": args.batch_size,
        "device": args.device,
        "c_grid": parse_c_grid(args.c_grid),
        "class_weight": args.class_weight,
        "use_oversampled_train": args.use_oversampled_train,
        "xrv_weights": args.xrv_weights,
        "rad_dino_model": args.rad_dino_model,
        "medsiglip_model": args.medsiglip_model,
        "label_mapping": {
            "0": "Abnormal",
            "1": "Normal",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained frozen embeddings on Normal/Abnormal CV folds",
    )
    parser.add_argument("--data-root", required=True, help="UniMiSSPlus data dir containing 2D_images/")
    parser.add_argument("--cv-dir", required=True, help="Directory produced by dicom_labeler.py build-cv-lists")
    parser.add_argument("--output-dir", default="results/pretrained_feature_eval", help="Output result directory")
    parser.add_argument(
        "--methods",
        default="rad_dino,torchxrayvision_densenet121,imagenet_densenet121,imagenet_resnet50",
        help="Comma-separated methods: rad_dino, torchxrayvision_densenet121, imagenet_densenet121, imagenet_resnet50, medsiglip",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=32, help="Feature extraction batch size")
    parser.add_argument("--c-grid", default="0.01,0.1,1,10,100", help="Logistic regression C values")
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Logistic regression class weighting; default relies on train-only oversampling",
    )
    parser.add_argument(
        "--use-oversampled-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use train_oversampled.txt for classifier fitting",
    )
    parser.add_argument("--xrv-weights", default="densenet121-res224-all", help="TorchXRayVision DenseNet weights")
    parser.add_argument("--rad-dino-model", default="microsoft/rad-dino", help="Hugging Face RAD-DINO model id")
    parser.add_argument("--medsiglip-model", default="google/medsiglip-448", help="Hugging Face model id")
    parser.add_argument("--no-cache", action="store_true", help="Do not read/write feature cache")
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="Continue when an optional model dependency is unavailable",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cv_dir = Path(args.cv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_methods(args.methods)
    folds = load_cv_folds(cv_dir)
    image_paths = unique_image_paths(folds)
    validate_image_paths(image_paths, data_root)
    write_run_config(output_dir / "feature_eval_run.json", args, methods)

    all_metrics = []
    all_predictions = []
    for method in methods:
        print(f"\n=== {method} ===")
        try:
            features_by_path = extract_features(method, image_paths, data_root, output_dir, args)
            metrics_rows, prediction_rows = evaluate_method(method, folds, features_by_path, args)
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            if args.skip_unavailable:
                print(f"  Skipped {method}: {exc}")
                continue
            raise
        all_metrics.extend(metrics_rows)
        all_predictions.extend(prediction_rows)

    metric_fields = [
        "method", "citation_key", "fold", "best_c", "val_auc_for_c",
        "val_average_precision_for_c", "train_list", "train_samples",
        "train_unique_images", "val_samples", "test_samples",
        "train_counts", "val_counts", "test_counts",
        "auc", "average_precision", "accuracy", "balanced_accuracy",
        "abnormal_recall", "specificity", "f1",
        "cm_abnormal_pred_abnormal", "cm_abnormal_pred_normal",
        "cm_normal_pred_abnormal", "cm_normal_pred_normal",
    ]
    prediction_fields = [
        "method", "fold", "image_path", "true_label", "true_label_name",
        "pred_label", "pred_label_name", "abnormal_score",
    ]
    write_csv(output_dir / "feature_eval_metrics.csv", all_metrics, metric_fields)
    write_csv(output_dir / "feature_eval_predictions.csv", all_predictions, prediction_fields)
    print(f"\nSaved metrics: {output_dir / 'feature_eval_metrics.csv'}")
    print(f"Saved pooled fold predictions: {output_dir / 'feature_eval_predictions.csv'}")


if __name__ == "__main__":
    main()
