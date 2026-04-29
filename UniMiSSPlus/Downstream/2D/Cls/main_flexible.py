import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import metrics
from torch.utils import data
from tqdm import tqdm

from dataset.flexible_datasets import (
    COVID_DEFAULT_CLASSES,
    COVID_OTHER_NORMAL_CLASSES,
    COVID_QU_EX_CLASSES,
    NIH_LABELS,
    CovidClassificationDataset,
    CovidOtherNormalDataset,
    CovidQUExClassificationDataset,
    NIHChestXrayDataset,
)
from net.MiTPlus_encoder import MiTPlus_encoder


def parse_args():
    parser = argparse.ArgumentParser("Flexible UniMiSS+ 2D downstream classification")
    parser.add_argument("--task", choices=["covid", "covid_qu_ex", "nih"], required=True)

    parser.add_argument("--covid_root", type=str, default=None,
                        help="Root of COVID-19_Radiography_Dataset. Expected class/images folders inside.")
    parser.add_argument("--covid_mode", choices=["selected", "other_normal"], default="selected",
                        help="'selected' uses --covid_classes. 'other_normal' maps COVID=0, Lung_Opacity+Viral Pneumonia=1, Normal=2.")
    parser.add_argument("--covid_classes", type=str, default=",".join(COVID_DEFAULT_CLASSES),
                        help="Comma-separated COVID classes to use. Default is 3-class COVID,Lung_Opacity,Normal.")
    parser.add_argument("--covid_train_list", type=str, default=None,
                        help="Optional labeled train list with '<relative_image_path> <0-based_label>'.")
    parser.add_argument("--covid_test_list", type=str, default=None,
                        help="Optional labeled test list with '<relative_image_path> <0-based_label>'.")
    parser.add_argument("--covid_test_split", type=float, default=0.2,
                        help="Used only when COVID train/test lists are not supplied.")
    parser.add_argument("--covid_qu_ex_root", type=str, default=None,
                        help="Root of COVID-QU-Ex, or nested 'Lung Segmentation Data/Lung Segmentation Data' folder.")
    parser.add_argument("--covid_qu_ex_subset", choices=["lung", "infection"], default="lung",
                        help="Which COVID-QU-Ex nested dataset to read. Both expose the same 3 classification folders plus masks.")
    parser.add_argument("--covid_qu_ex_train_splits", type=str, default="Train,Val",
                        help="Comma-separated COVID-QU-Ex split folders used for training.")
    parser.add_argument("--covid_qu_ex_test_splits", type=str, default="Test",
                        help="Comma-separated COVID-QU-Ex split folders used for testing.")

    parser.add_argument("--nih_root", type=str, default=None,
                        help="Root of NIH CXR folder. Expected images_*/images folders inside.")
    parser.add_argument("--nih_csv", type=str, default=None,
                        help="Path to NIH Data_Entry_2017.csv.")
    parser.add_argument("--nih_train_list", type=str, default=None,
                        help="Path to NIH train_val_list.txt or another filename list.")
    parser.add_argument("--nih_test_list", type=str, default=None,
                        help="Path to NIH test_list.txt or another filename list.")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pre_train", action="store_true")
    parser.add_argument("--pre_train_path", type=str, default=None)
    parser.add_argument("--pre_type", choices=["student", "teacher"], default="student")

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use torch.nn.DataParallel when multiple visible GPUs are available.")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", choices=["poly", "cosine", "constant"], default="poly",
                        help="Learning-rate schedule. 'poly' matches the original downstream code style.")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="Polynomial LR power, used when --lr_schedule poly.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum LR for cosine schedule.")
    parser.add_argument("--weight_decay", type=float, default=3e-5)
    parser.add_argument("--optimizer", choices=["AdamW", "Adam", "SGD"], default="AdamW")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--normalize", choices=["chestx-ray", "imagenet", "none"], default="chestx-ray")
    parser.add_argument("--test_augment", action="store_true",
                        help="Use TenCrop at test time. Slower, but matches the original downstream script.")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Model state_dict to evaluate or fine-tune from.")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Full training checkpoint to resume from. Restores model, optimizer, epoch and best score.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable PNG/TXT/NPZ evaluation artifacts.")
    parser.add_argument("--plot_dpi", type=int, default=160)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_unimiss_pretrained(model, checkpoint_path, pre_type="student"):
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if pre_type not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain key '{pre_type}'")

    state = checkpoint[pre_type]
    if pre_type == "teacher":
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
    else:
        state = {k.replace("module.backbone.", ""): v for k, v in state.items()}
    state = {k.replace("transformer.", ""): v for k, v in state.items()}

    model_state = model.state_dict()
    for key, value in list(state.items()):
        if "pos_embed2D" in key and key in model_state and value.size() != model_state[key].size():
            target_tokens = model_state[key].size(1)
            resized = ndimage.zoom(value[0].numpy(), (target_tokens / value.size(1), 1), order=1)
            state[key] = torch.from_numpy(np.expand_dims(resized, 0))

    matched = {k: v for k, v in state.items() if k in model_state and v.size() == model_state[k].size()}
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"Loaded {len(matched)} pretrained layers from {checkpoint_path}")


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_model_weights(model, checkpoint_path):
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    unwrap_model(model).load_state_dict(checkpoint)
    print(f"Loaded model weights from {checkpoint_path}")


def load_training_checkpoint(model, optimizer, resume_path):
    checkpoint = safe_torch_load(resume_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            f"{resume_path} is not a full training checkpoint. "
            "Use --checkpoint_path for plain model weights."
        )
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        move_optimizer_state_to_device(optimizer, next(model.parameters()).device)
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_score = float(checkpoint.get("best_score", -1.0))
    print(f"Resumed training from {resume_path} at epoch {start_epoch}")
    return start_epoch, best_score


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(path, model, optimizer, epoch, best_score, args, label_names):
    torch.save({
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "args": vars(args),
        "label_names": list(label_names),
    }, path)


def build_loaders(args):
    if args.task == "covid":
        if not args.covid_root:
            raise ValueError("--covid_root is required for --task covid")
        if args.covid_mode == "other_normal":
            train_set = CovidOtherNormalDataset(
                args.covid_root, "train", args.covid_train_list, args.covid_test_list,
                args.covid_test_split, args.seed, args.input_size, args.resize, args.normalize, False)
            test_set = CovidOtherNormalDataset(
                args.covid_root, "test", args.covid_train_list, args.covid_test_list,
                args.covid_test_split, args.seed, args.input_size, args.resize, args.normalize, args.test_augment)
            label_names = list(COVID_OTHER_NORMAL_CLASSES)
        else:
            classes = tuple(item.strip() for item in args.covid_classes.split(",") if item.strip())
            train_set = CovidClassificationDataset(
                args.covid_root, "train", classes, args.covid_train_list, args.covid_test_list,
                args.covid_test_split, args.seed, args.input_size, args.resize, args.normalize, False)
            test_set = CovidClassificationDataset(
                args.covid_root, "test", classes, args.covid_train_list, args.covid_test_list,
                args.covid_test_split, args.seed, args.input_size, args.resize, args.normalize, args.test_augment)
            label_names = list(classes)
        num_classes = len(label_names)
    elif args.task == "covid_qu_ex":
        if not args.covid_qu_ex_root:
            raise ValueError("--covid_qu_ex_root is required for --task covid_qu_ex")
        train_splits = tuple(item.strip() for item in args.covid_qu_ex_train_splits.split(",") if item.strip())
        test_splits = tuple(item.strip() for item in args.covid_qu_ex_test_splits.split(",") if item.strip())
        train_set = CovidQUExClassificationDataset(
            args.covid_qu_ex_root, "train", args.covid_qu_ex_subset, train_splits, test_splits,
            args.input_size, args.resize, args.normalize, False)
        test_set = CovidQUExClassificationDataset(
            args.covid_qu_ex_root, "test", args.covid_qu_ex_subset, train_splits, test_splits,
            args.input_size, args.resize, args.normalize, args.test_augment)
        num_classes = len(COVID_QU_EX_CLASSES)
        label_names = list(COVID_QU_EX_CLASSES)
    else:
        required = [args.nih_root, args.nih_csv, args.nih_train_list, args.nih_test_list]
        if any(value is None for value in required):
            raise ValueError("--nih_root, --nih_csv, --nih_train_list and --nih_test_list are required for --task nih")
        train_set = NIHChestXrayDataset(
            args.nih_root, args.nih_csv, args.nih_train_list, "train",
            args.input_size, args.resize, args.normalize, False)
        test_set = NIHChestXrayDataset(
            args.nih_root, args.nih_csv, args.nih_test_list, "test",
            args.input_size, args.resize, args.normalize, args.test_augment)
        num_classes = len(NIH_LABELS)
        label_names = list(NIH_LABELS)

    train_loader = data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False)
    test_loader = data.DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False)
    return train_loader, test_loader, num_classes, label_names


def make_optimizer(args, model):
    if args.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               momentum=0.99, nesterov=True)
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def compute_learning_rate(args, step, total_steps):
    if args.lr_schedule == "constant":
        return args.learning_rate
    progress = min(float(step) / max(1, total_steps), 1.0)
    if args.lr_schedule == "poly":
        return args.learning_rate * ((1.0 - progress) ** args.lr_power)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return args.min_lr + (args.learning_rate - args.min_lr) * cosine


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def format_metrics_line(result):
    ordered_keys = [
        "epoch",
        "train_loss",
        "lr",
        "accuracy",
        "macro_auc",
        "macro_ap",
        "mean_auc",
        "mean_ap",
        "micro_f1",
        "exact_match_accuracy",
        "label_accuracy",
    ]
    parts = []
    for key in ordered_keys:
        if key not in result:
            continue
        value = result[key]
        if value is None:
            parts.append(f"{key}=NA")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _plot_confusion_matrix(y_true, y_pred, label_names, output_dir, prefix, dpi):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 1.4), max(5, len(label_names) * 1.2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=dpi)
    plt.close(fig)
    np.savetxt(output_dir / f"{prefix}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")


def _plot_single_label_curves(y_true, y_score, label_names, output_dir, prefix, dpi):
    y_true_one_hot = np.eye(len(label_names))[y_true.astype(np.int64)]

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, name in enumerate(label_names):
        if len(np.unique(y_true_one_hot[:, idx])) < 2:
            continue
        fpr, tpr, _ = metrics.roc_curve(y_true_one_hot[:, idx], y_score[:, idx])
        auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc_curves.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, name in enumerate(label_names):
        precision, recall, _ = metrics.precision_recall_curve(y_true_one_hot[:, idx], y_score[:, idx])
        ap = metrics.average_precision_score(y_true_one_hot[:, idx], y_score[:, idx])
        ax.plot(recall, precision, label=f"{name} AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_pr_curves.png", dpi=dpi)
    plt.close(fig)


def _plot_bar(values, label_names, title, ylabel, path, dpi):
    numeric = [np.nan if value is None else float(value) for value in values]
    fig, ax = plt.subplots(figsize=(max(8, len(label_names) * 0.55), 5))
    ax.bar(np.arange(len(label_names)), numeric)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_multilabel_artifacts(y_true, y_score, label_names, result, output_dir, prefix, dpi):
    y_pred = (y_score >= 0.5).astype(np.float32)
    with open(output_dir / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(metrics.classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    per_auc = result.get("per_class_auc", {})
    per_ap = result.get("per_class_ap", {})
    _plot_bar([per_auc.get(name) for name in label_names], label_names, "Per-Class ROC-AUC", "ROC-AUC",
              output_dir / f"{prefix}_per_class_auc.png", dpi)
    _plot_bar([per_ap.get(name) for name in label_names], label_names, "Per-Class Average Precision", "AP",
              output_dir / f"{prefix}_per_class_ap.png", dpi)

    positives = y_true.sum(axis=0)
    fig, ax = plt.subplots(figsize=(max(8, len(label_names) * 0.55), 5))
    ax.bar(np.arange(len(label_names)), positives)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_ylabel("Positive samples")
    ax.set_title("Evaluation Label Frequency")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_label_frequency.png", dpi=dpi)
    plt.close(fig)

    flat_true = y_true.reshape(-1)
    flat_score = y_score.reshape(-1)
    if len(np.unique(flat_true)) >= 2:
        fpr, tpr, _ = metrics.roc_curve(flat_true, flat_score)
        precision, recall, _ = metrics.precision_recall_curve(flat_true, flat_score)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f"micro AUC={metrics.auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Micro-Averaged ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_micro_roc.png", dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, label=f"micro AP={metrics.average_precision_score(flat_true, flat_score):.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Micro-Averaged Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_micro_pr.png", dpi=dpi)
        plt.close(fig)


def save_eval_artifacts(y_true, y_score, result, task, label_names, output_dir, prefix, dpi):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{prefix}_predictions.npz",
        y_true=y_true,
        y_score=y_score,
        label_names=np.array(label_names),
    )
    _save_json(output_dir / f"{prefix}_metrics.json", result)

    if task in ("covid", "covid_qu_ex"):
        y_pred = y_score.argmax(axis=1)
        with open(output_dir / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(metrics.classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
        _plot_confusion_matrix(y_true, y_pred, label_names, output_dir, prefix, dpi)
        _plot_single_label_curves(y_true, y_score, label_names, output_dir, prefix, dpi)
    else:
        _plot_multilabel_artifacts(y_true, y_score, label_names, result, output_dir, prefix, dpi)


def plot_history(history_path, output_dir, task, dpi):
    history_path = Path(history_path)
    if not history_path.is_file():
        return
    rows = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    epochs = [row["epoch"] for row in rows]
    metric_names = ["train_loss", "lr"]
    if task in ("covid", "covid_qu_ex"):
        metric_names.extend(["accuracy", "macro_auc", "macro_ap"])
    else:
        metric_names.extend(["mean_auc", "mean_ap", "micro_f1", "exact_match_accuracy", "label_accuracy"])

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(8, max(3 * len(metric_names), 4)), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]
    for ax, metric_name in zip(axes, metric_names):
        values = [row.get(metric_name) for row in rows]
        if all(value is None for value in values):
            ax.text(0.5, 0.5, f"No {metric_name}", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(epochs, values, marker="o")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curves.png", dpi=dpi)
    plt.close(fig)


def forward_eval(model, images, task):
    if images.ndim == 5:
        batch_size, crops, channels, height, width = images.size()
        flat = images.view(-1, channels, height, width)
        logits = model(flat).view(batch_size, crops, -1).mean(1)
    else:
        logits = model(images)
    if task in ("covid", "covid_qu_ex"):
        return torch.softmax(logits, dim=1)
    return torch.sigmoid(logits)


def evaluate(model, loader, task, device, label_names, output_dir=None, save_plots=True, plot_dpi=160,
             artifact_prefix="eval"):
    model.eval()
    targets = []
    scores = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval"):
            images = images.to(device, non_blocking=True)
            probs = forward_eval(model, images, task)
            scores.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    y_score = np.concatenate(scores, axis=0)
    y_true = np.concatenate(targets, axis=0)

    if task in ("covid", "covid_qu_ex"):
        y_pred = y_score.argmax(axis=1)
        acc = metrics.accuracy_score(y_true, y_pred)
        y_true_one_hot = np.eye(len(label_names))[y_true.astype(np.int64)]
        result = {"accuracy": float(acc)}
        try:
            result["macro_auc"] = float(metrics.roc_auc_score(y_true_one_hot, y_score, average="macro", multi_class="ovr"))
        except ValueError:
            result["macro_auc"] = None
        result["macro_ap"] = float(metrics.average_precision_score(y_true_one_hot, y_score, average="macro"))
        if save_plots and output_dir is not None:
            save_eval_artifacts(y_true, y_score, result, task, label_names, output_dir, artifact_prefix, plot_dpi)
        return result

    y_pred = (y_score >= 0.5).astype(np.float32)
    per_class_auc = {}
    per_class_ap = {}
    valid_aucs = []
    valid_aps = []
    for idx, name in enumerate(label_names):
        if len(np.unique(y_true[:, idx])) < 2:
            per_class_auc[name] = None
        else:
            auc = float(metrics.roc_auc_score(y_true[:, idx], y_score[:, idx]))
            per_class_auc[name] = auc
            valid_aucs.append(auc)
        ap = float(metrics.average_precision_score(y_true[:, idx], y_score[:, idx]))
        per_class_ap[name] = ap
        valid_aps.append(ap)

    result = {
        "mean_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
        "mean_ap": float(np.mean(valid_aps)) if valid_aps else None,
        "micro_f1": float(metrics.f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0)),
        "exact_match_accuracy": float(np.mean(np.all(y_pred == y_true, axis=1))),
        "label_accuracy": float(np.mean(y_pred == y_true)),
        "per_class_auc": per_class_auc,
        "per_class_ap": per_class_ap,
    }
    if save_plots and output_dir is not None:
        save_eval_artifacts(y_true, y_score, result, task, label_names, output_dir, artifact_prefix, plot_dpi)
    return result


def train_one_epoch(model, loader, criterion, optimizer, task, device, epoch, args, total_steps):
    model.train()
    losses = []
    last_lr = optimizer.param_groups[0]["lr"]
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"train {epoch}")):
        global_step = epoch * len(loader) + batch_idx
        last_lr = compute_learning_rate(args, global_step, total_steps)
        set_optimizer_lr(optimizer, last_lr)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if task in ("covid", "covid_qu_ex"):
            labels = labels.long()
        else:
            labels = labels.float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)), float(last_lr)


def main():
    args = parse_args()
    if args.checkpoint_path and args.resume_path:
        raise ValueError("Use either --checkpoint_path or --resume_path, not both.")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, num_classes, label_names = build_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiTPlus_encoder(num_classes=num_classes)
    print("  + Number of Network Params: %.2f(e6)" % (sum(p.nelement() for p in model.parameters()) / 1e6))

    if args.pre_train:
        if not args.pre_train_path:
            raise ValueError("--pre_train_path is required when --pre_train is set")
        load_unimiss_pretrained(model, args.pre_train_path, args.pre_type)
    if args.checkpoint_path:
        load_model_weights(model, args.checkpoint_path)

    model.to(device).float()
    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1 and (args.multi_gpu or "," in args.gpu)
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    elif args.multi_gpu:
        print("--multi_gpu was set, but fewer than 2 CUDA GPUs are visible; using single-device training")
    criterion = torch.nn.CrossEntropyLoss() if args.task in ("covid", "covid_qu_ex") else torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    optimizer = make_optimizer(args, model)
    start_epoch = 0
    best_score = -1.0
    if args.resume_path:
        start_epoch, best_score = load_training_checkpoint(model, optimizer, args.resume_path)

    history_path = output_dir / "metrics.jsonl"
    save_plots = not args.no_plots
    if args.eval_only:
        result = evaluate(
            model, test_loader, args.task, device, label_names,
            output_dir=output_dir, save_plots=save_plots, plot_dpi=args.plot_dpi,
            artifact_prefix="eval",
        )
        print(format_metrics_line(result))
        return

    total_steps = len(train_loader) * args.epochs
    if start_epoch >= args.epochs:
        print(f"Nothing to train: resume epoch {start_epoch} >= --epochs {args.epochs}")
        return
    for epoch in range(start_epoch, args.epochs):
        train_loss, lr = train_one_epoch(model, train_loader, criterion, optimizer, args.task, device, epoch, args, total_steps)
        result = evaluate(
            model, test_loader, args.task, device, label_names,
            output_dir=output_dir, save_plots=save_plots, plot_dpi=args.plot_dpi,
            artifact_prefix="latest_eval",
        )
        result["epoch"] = epoch
        result["train_loss"] = train_loss
        result["lr"] = lr
        print(format_metrics_line(result))

        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        if save_plots:
            plot_history(history_path, output_dir, args.task, args.plot_dpi)

        torch.save(unwrap_model(model).state_dict(), output_dir / "last.pth")
        score = result["accuracy"] if args.task in ("covid", "covid_qu_ex") else result["mean_auc"]
        if score is not None and score > best_score:
            best_score = score
            torch.save(unwrap_model(model).state_dict(), output_dir / "best.pth")
            _save_json(output_dir / "best_metrics.json", result)
        save_training_checkpoint(output_dir / "checkpoint.pth", model, optimizer, epoch, best_score, args, label_names)


if __name__ == "__main__":
    main()

