import argparse
import gc
import json
import os
import random
import shutil
import time
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
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds to run sequentially, e.g. '1234,2024,42'. Each run writes to <output_dir>/seed_<seed>.")
    parser.add_argument("--loss", choices=["ce", "asl"], default="ce",
                        help="Loss for single-label covid tasks: 'ce' cross-entropy, 'asl' single-label Asymmetric Loss for class imbalance.")
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0, help="ASL focusing for negatives (majority class).")
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0, help="ASL focusing for positives.")
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="none",
                        help="'balanced' weights CE loss by inverse class frequency of --covid_train_list (for imbalanced covid tasks).")
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
    parser.add_argument("--export_onnx", action="store_true",
                        help="Export an ONNX model after eval-only or after training finishes.")
    parser.add_argument("--onnx_path", type=str, default=None,
                        help="Output ONNX path. Defaults to <output_dir>/best.onnx after training or <output_dir>/eval.onnx in eval-only.")
    parser.add_argument("--onnx_opset", type=int, default=17,
                        help="ONNX opset version to use for export.")
    parser.add_argument("--onnx_export_weights", choices=["best", "last", "current"], default="best",
                        help="Which weights to export after training. Eval-only always exports the loaded current model.")
    parser.add_argument("--benchmark_inference", action="store_true",
                        help="Measure PyTorch inference speed on the test loader.")
    parser.add_argument("--compare_onnx", action="store_true",
                        help="Compare PyTorch and ONNX Runtime accuracy, outputs, and inference speed on the test loader.")
    parser.add_argument("--benchmark_warmup_batches", type=int, default=2,
                        help="Number of initial batches to exclude from speed timing.")
    parser.add_argument("--benchmark_max_batches", type=int, default=0,
                        help="Maximum benchmark batches. 0 means use the full test loader.")
    parser.add_argument("--onnx_runtime_provider", choices=["auto", "cpu", "cuda"], default="auto",
                        help="ONNX Runtime provider preference for comparison/benchmarking.")
    parser.add_argument("--onnx_batch_size", type=int, default=32,
                        help="Chunk size used only for ONNX Runtime evaluation/benchmarking. Use 0 to run full DataLoader batches.")
    parser.add_argument("--benchmark_output", type=str, default=None,
                        help="JSON path for benchmark/ONNX comparison results. Defaults inside output_dir.")
    parser.add_argument("--grad_cam", action="store_true",
                        help="Save Grad-CAM overlays for samples from the test loader.")
    parser.add_argument("--grad_cam_samples", type=int, default=8,
                        help="Number of test samples to visualize with Grad-CAM.")
    parser.add_argument("--grad_cam_per_class", action="store_true",
                        help="For single-label tasks, save a balanced set of Grad-CAM overlays by true class.")
    parser.add_argument("--grad_cam_per_class_samples", type=int, default=1,
                        help="Number of Grad-CAM overlays to save for each class when --grad_cam_per_class is set.")
    parser.add_argument("--grad_cam_layer", type=str, default="patch_embed2D4.proj1.conv",
                        help="Target module name for Grad-CAM. Falls back to patch_embed2D4.proj.conv if missing.")
    parser.add_argument("--grad_cam_class", type=int, default=None,
                        help="Class index to explain. Defaults to predicted class for each sample.")
    parser.add_argument("--grad_cam_dir", type=str, default=None,
                        help="Directory for Grad-CAM images. Defaults to <output_dir>/grad_cam.")
    parser.add_argument("--grad_cam_weights", choices=["best", "last", "current"], default="best",
                        help="Which weights to use for Grad-CAM after training. Eval-only always uses the loaded current model.")
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


def export_onnx_model(model, output_path, input_size, device, opset_version=17):
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the 'onnx' Python package. "
            "Install it first, for example: pip install onnx"
        ) from exc

    export_model = unwrap_model(model).to(device).float()
    was_training = export_model.training
    export_model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device, dtype=torch.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch"},
                "logits": {0: "batch"},
            },
            dynamo=False,
        )
    finally:
        if was_training:
            export_model.train()
    print(f"Exported ONNX model to {output_path}")


def export_requested_onnx(model, args, output_dir, device, eval_only=False):
    if not args.export_onnx:
        return None

    if args.onnx_path:
        onnx_path = Path(args.onnx_path)
    else:
        onnx_path = output_dir / ("eval.onnx" if eval_only else f"{args.onnx_export_weights}.onnx")

    if not eval_only and args.onnx_export_weights in ("best", "last"):
        weights_path = output_dir / f"{args.onnx_export_weights}.pth"
        if weights_path.exists():
            load_model_weights(model, weights_path)
        else:
            print(f"ONNX export requested {args.onnx_export_weights}.pth, but it was not found; exporting current model.")

    export_onnx_model(model, onnx_path, args.input_size, device, args.onnx_opset)
    return onnx_path


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


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _speed_summary(seconds, samples, batches):
    if seconds <= 0 or samples <= 0:
        return {
            "timed_seconds": float(seconds),
            "timed_samples": int(samples),
            "timed_batches": int(batches),
            "samples_per_second": None,
            "milliseconds_per_image": None,
        }
    return {
        "timed_seconds": float(seconds),
        "timed_samples": int(samples),
        "timed_batches": int(batches),
        "samples_per_second": float(samples / seconds),
        "milliseconds_per_image": float(seconds * 1000.0 / samples),
    }


def collect_pytorch_predictions(model, loader, task, device, desc="eval", warmup_batches=0, max_batches=0):
    targets = []
    scores = []
    timed_seconds = 0.0
    timed_samples = 0
    timed_batches = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=desc)):
            if max_batches and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            do_time = batch_idx >= warmup_batches
            if do_time:
                _sync_device(device)
                start = time.perf_counter()
            probs = forward_eval(model, images, task)
            if do_time:
                _sync_device(device)
                timed_seconds += time.perf_counter() - start
                timed_samples += int(images.size(0))
                timed_batches += 1
            scores.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    return (
        np.concatenate(targets, axis=0),
        np.concatenate(scores, axis=0),
        _speed_summary(timed_seconds, timed_samples, timed_batches),
    )


def compute_eval_metrics(y_true, y_score, task, label_names):
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
    return result


def evaluate(model, loader, task, device, label_names, output_dir=None, save_plots=True, plot_dpi=160,
             artifact_prefix="eval"):
    y_true, y_score, _ = collect_pytorch_predictions(model, loader, task, device, desc="eval")
    result = compute_eval_metrics(y_true, y_score, task, label_names)
    if save_plots and output_dir is not None:
        save_eval_artifacts(y_true, y_score, result, task, label_names, output_dir, artifact_prefix, plot_dpi)
    return result


def _softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _sigmoid_np(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def create_onnx_session(onnx_path, provider_preference="auto"):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX comparison requires the 'onnxruntime' Python package. "
            "Install it first, for example: pip install onnxruntime"
        ) from exc

    if provider_preference != "cpu" and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True)
        except Exception as exc:
            print(f"ONNX Runtime CUDA preload warning: {exc}")

    available = ort.get_available_providers()
    if provider_preference == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider_preference == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in onnxruntime. "
                "Install onnxruntime-gpu or use --onnx_runtime_provider cpu."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_providers = session.get_providers()
    if provider_preference == "cuda" and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "ONNX Runtime CUDA provider was requested but could not be activated. "
            "This usually means the installed onnxruntime-gpu build does not match the local CUDA/cuDNN libraries. "
            "Use --onnx_runtime_provider cpu, or install an onnxruntime-gpu build and CUDA/cuDNN runtime that match each other."
        )
    print(f"Loaded ONNX Runtime session from {onnx_path} with providers={active_providers}")
    return session


def forward_onnx_eval(session, images, task, onnx_batch_size=32):
    input_name = session.get_inputs()[0].name

    def run_chunks(array):
        array = array.astype(np.float32)
        chunk_size = int(onnx_batch_size or 0)
        if chunk_size <= 0 or array.shape[0] <= chunk_size:
            return session.run(None, {input_name: array})[0]
        outputs = []
        for start in range(0, array.shape[0], chunk_size):
            outputs.append(session.run(None, {input_name: array[start:start + chunk_size]})[0])
        return np.concatenate(outputs, axis=0)

    if images.ndim == 5:
        batch_size, crops, channels, height, width = images.shape
        flat = images.reshape(batch_size * crops, channels, height, width)
        logits = run_chunks(flat)
        logits = logits.reshape(batch_size, crops, -1).mean(axis=1)
    else:
        logits = run_chunks(images)
    if task in ("covid", "covid_qu_ex"):
        return _softmax_np(logits)
    return _sigmoid_np(logits)


def collect_onnx_predictions(session, loader, task, desc="onnx eval", warmup_batches=0, max_batches=0, onnx_batch_size=32):
    targets = []
    scores = []
    timed_seconds = 0.0
    timed_samples = 0
    timed_batches = 0
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=desc)):
        if max_batches and batch_idx >= max_batches:
            break
        images_np = images.cpu().numpy()
        do_time = batch_idx >= warmup_batches
        if do_time:
            start = time.perf_counter()
        probs = forward_onnx_eval(session, images_np, task, onnx_batch_size)
        if do_time:
            timed_seconds += time.perf_counter() - start
            timed_samples += int(images.size(0))
            timed_batches += 1
        scores.append(probs)
        targets.append(labels.cpu().numpy())

    return (
        np.concatenate(targets, axis=0),
        np.concatenate(scores, axis=0),
        _speed_summary(timed_seconds, timed_samples, timed_batches),
    )


def compare_predictions(y_true, pytorch_score, onnx_score, task):
    diff = np.abs(pytorch_score - onnx_score)
    result = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }
    if task in ("covid", "covid_qu_ex"):
        pt_pred = pytorch_score.argmax(axis=1)
        onnx_pred = onnx_score.argmax(axis=1)
        result["prediction_agreement"] = float(np.mean(pt_pred == onnx_pred))
        result["pytorch_accuracy"] = float(metrics.accuracy_score(y_true, pt_pred))
        result["onnx_accuracy"] = float(metrics.accuracy_score(y_true, onnx_pred))
        result["accuracy_delta"] = float(result["onnx_accuracy"] - result["pytorch_accuracy"])
    else:
        pt_pred = (pytorch_score >= 0.5).astype(np.float32)
        onnx_pred = (onnx_score >= 0.5).astype(np.float32)
        result["label_prediction_agreement"] = float(np.mean(pt_pred == onnx_pred))
        result["exact_prediction_agreement"] = float(np.mean(np.all(pt_pred == onnx_pred, axis=1)))
    return result


def ensure_onnx_for_comparison(model, args, output_dir, device, eval_only=False):
    if args.onnx_path:
        onnx_path = Path(args.onnx_path)
    else:
        onnx_path = output_dir / ("eval.onnx" if eval_only else f"{args.onnx_export_weights}.onnx")

    should_export = args.export_onnx or not onnx_path.exists()
    if not should_export:
        return onnx_path

    if not eval_only and args.onnx_export_weights in ("best", "last"):
        weights_path = output_dir / f"{args.onnx_export_weights}.pth"
        if weights_path.exists():
            load_model_weights(model, weights_path)
        else:
            print(f"ONNX comparison requested {args.onnx_export_weights}.pth, but it was not found; exporting current model.")

    export_onnx_model(model, onnx_path, args.input_size, device, args.onnx_opset)
    return onnx_path


def run_inference_benchmark(model, loader, task, device, label_names, args, output_dir,
                            eval_only=False, pytorch_result=None):
    max_batches = args.benchmark_max_batches if args.benchmark_max_batches > 0 else 0
    payload = {}
    onnx_path = None
    session = None

    if args.compare_onnx:
        onnx_path = ensure_onnx_for_comparison(model, args, output_dir, device, eval_only=eval_only)
        session = create_onnx_session(onnx_path, args.onnx_runtime_provider)

    if args.benchmark_inference or args.compare_onnx:
        y_true, pt_score, pt_speed = collect_pytorch_predictions(
            model, loader, task, device,
            desc="pytorch benchmark",
            warmup_batches=args.benchmark_warmup_batches,
            max_batches=max_batches,
        )
        payload["pytorch"] = {
            "metrics": compute_eval_metrics(y_true, pt_score, task, label_names),
            "speed": pt_speed,
        }
    else:
        y_true, pt_score = None, None

    if args.compare_onnx:
        onnx_true, onnx_score, onnx_speed = collect_onnx_predictions(
            session, loader, task,
            desc="onnx benchmark",
            warmup_batches=args.benchmark_warmup_batches,
            max_batches=max_batches,
            onnx_batch_size=args.onnx_batch_size,
        )
        if y_true is None or pt_score is None:
            y_true, pt_score, pt_speed = collect_pytorch_predictions(
                model, loader, task, device,
                desc="pytorch benchmark",
                warmup_batches=args.benchmark_warmup_batches,
                max_batches=max_batches,
            )
            payload["pytorch"] = {
                "metrics": compute_eval_metrics(y_true, pt_score, task, label_names),
                "speed": pt_speed,
            }
        if not np.array_equal(y_true, onnx_true):
            raise RuntimeError("PyTorch and ONNX benchmark loaders produced different target order.")
        payload["onnx"] = {
            "path": str(onnx_path),
            "providers": session.get_providers(),
            "metrics": compute_eval_metrics(onnx_true, onnx_score, task, label_names),
            "speed": onnx_speed,
        }
        payload["comparison"] = compare_predictions(y_true, pt_score, onnx_score, task)

    if pytorch_result is not None and "pytorch" in payload:
        payload["pytorch"]["full_eval_metrics"] = pytorch_result

    if payload:
        output_path = Path(args.benchmark_output) if args.benchmark_output else output_dir / "inference_benchmark.json"
        _save_json(output_path, payload)
        print(f"Saved inference benchmark to {output_path}")
        if "pytorch" in payload:
            print("PyTorch benchmark:", json.dumps(payload["pytorch"]["speed"]))
        if "onnx" in payload:
            print("ONNX benchmark:", json.dumps(payload["onnx"]["speed"]))
            print("PyTorch vs ONNX:", json.dumps(payload["comparison"]))
    return payload


def normalization_stats(normalize):
    normalize = normalize.lower()
    if normalize == "imagenet":
        return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    if normalize == "chestx-ray":
        return np.array([0.5056, 0.5056, 0.5056]), np.array([0.252, 0.252, 0.252])
    return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])


def tensor_to_image_array(image_tensor, normalize):
    image = image_tensor.detach().cpu().float().numpy()
    image = np.transpose(image, (1, 2, 0))
    mean, std = normalization_stats(normalize)
    image = image * std + mean
    return np.clip(image, 0.0, 1.0)


def get_module_by_name(model, module_name):
    modules = dict(unwrap_model(model).named_modules())
    if module_name in modules:
        return modules[module_name], module_name
    fallback = "patch_embed2D4.proj.conv"
    if fallback in modules:
        print(f"Grad-CAM layer '{module_name}' was not found; using '{fallback}' instead.")
        return modules[fallback], fallback
    preview = ", ".join(list(modules.keys())[:20])
    raise ValueError(f"Grad-CAM layer '{module_name}' was not found. First modules: {preview}")


def compute_grad_cam(model, image, task, target_class, target_layer_name):
    model = unwrap_model(model)
    model.eval()
    target_layer, resolved_layer_name = get_module_by_name(model, target_layer_name)
    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output

    def backward_hook(_module, _grad_input, grad_output):
        captured["gradient"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        if target_class is None:
            if task in ("covid", "covid_qu_ex"):
                class_idx = int(logits.argmax(dim=1).item())
            else:
                class_idx = int(torch.sigmoid(logits).argmax(dim=1).item())
        else:
            class_idx = int(target_class)
        score = logits[:, class_idx].sum()
        score.backward()

        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        probs = torch.softmax(logits, dim=1) if task in ("covid", "covid_qu_ex") else torch.sigmoid(logits)
        return cam.cpu().numpy(), int(class_idx), float(probs[0, class_idx].detach().cpu()), resolved_layer_name
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_grad_cam_overlay(base_image, cam, output_path, title=None):
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * base_image + 0.45 * heatmap, 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(base_image)
    axes[0].set_title("Image")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_grad_cam(model, loader, task, device, label_names, args, output_dir, eval_only=False):
    if not args.grad_cam:
        return

    if args.grad_cam_per_class and task not in ("covid", "covid_qu_ex"):
        raise ValueError("--grad_cam_per_class is only supported for single-label tasks: covid and covid_qu_ex")

    if not eval_only and args.grad_cam_weights in ("best", "last"):
        weights_path = Path(output_dir) / f"{args.grad_cam_weights}.pth"
        if weights_path.exists():
            load_model_weights(model, weights_path)
        else:
            print(f"Grad-CAM requested {args.grad_cam_weights}.pth, but it was not found; using current model.")

    grad_cam_dir = Path(args.grad_cam_dir) if args.grad_cam_dir else Path(output_dir) / "grad_cam"
    grad_cam_dir.mkdir(parents=True, exist_ok=True)
    model = unwrap_model(model).to(device).float()
    saved = []
    sample_idx = 0

    if args.grad_cam_per_class:
        per_class_limit = max(1, int(args.grad_cam_per_class_samples))
        saved_per_class = {class_idx: 0 for class_idx in range(len(label_names))}
        total_target = per_class_limit * len(label_names)
    else:
        per_class_limit = None
        saved_per_class = None
        total_target = max(0, int(args.grad_cam_samples))

    for images, labels in tqdm(loader, desc="grad-cam"):
        if sample_idx >= total_target:
            break
        if images.ndim == 5:
            images = images[:, 0]
        images = images.to(device, non_blocking=True)
        for item_idx in range(images.size(0)):
            if sample_idx >= total_target:
                break

            label_value = labels[item_idx].detach().cpu()
            true_label = None
            true_name = None
            target_class = args.grad_cam_class
            if task in ("covid", "covid_qu_ex"):
                true_label = int(label_value.item())
                true_name = label_names[true_label]
                if args.grad_cam_per_class:
                    if saved_per_class[true_label] >= per_class_limit:
                        continue
                    target_class = true_label
            else:
                positive = [label_names[i] for i, value in enumerate(label_value.numpy()) if value > 0.5]
                true_name = "|".join(positive) if positive else "None"

            image = images[item_idx:item_idx + 1].clone().detach().requires_grad_(True)
            cam, class_idx, confidence, layer_name = compute_grad_cam(
                model, image, task, target_class, args.grad_cam_layer
            )
            base_image = tensor_to_image_array(image[0], args.normalize)
            pred_name = label_names[class_idx]
            if args.grad_cam_per_class:
                output_path = grad_cam_dir / f"grad_cam_true_{true_label:02d}_{true_name}_{saved_per_class[true_label]:02d}.png"
            else:
                output_path = grad_cam_dir / f"grad_cam_{sample_idx:04d}_class_{class_idx}.png"
            title = f"target={pred_name} conf={confidence:.4f} true={true_name}"
            save_grad_cam_overlay(base_image, cam, output_path, title)
            saved.append({
                "path": str(output_path),
                "target_class": class_idx,
                "target_name": pred_name,
                "confidence": confidence,
                "true_label": true_name,
                "layer": layer_name,
            })
            if args.grad_cam_per_class:
                saved[-1]["true_class"] = true_label
                saved_per_class[true_label] += 1
            sample_idx += 1

    if args.grad_cam_per_class:
        missing = [label_names[class_idx] for class_idx, count in saved_per_class.items() if count < per_class_limit]
        if missing:
            print(f"Grad-CAM per-class warning: missing samples for {missing}")

    _save_json(grad_cam_dir / "grad_cam_summary.json", saved)
    print(f"Saved {len(saved)} Grad-CAM overlays to {grad_cam_dir}")

def parse_seed_list(seeds_text):
    seeds = []
    for item in seeds_text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def primary_score_key(task):
    return "accuracy" if task in ("covid", "covid_qu_ex") else "mean_auc"


def promote_best_seed(seed_results, output_dir, task):
    if not seed_results:
        return None
    score_key = primary_score_key(task)
    valid_runs = [run for run in seed_results if run.get("metrics", {}).get(score_key) is not None]
    if not valid_runs:
        return None

    best_run = max(valid_runs, key=lambda run: float(run["metrics"][score_key]))
    best_run["score_key"] = score_key
    best_run["score"] = float(best_run["metrics"][score_key])

    output_dir = Path(output_dir)
    best_seed_dir = Path(best_run["output_dir"])
    for name in ("best.pth", "best_metrics.json", "checkpoint.pth", "last.pth", "best.onnx", "inference_benchmark.json"):
        source = best_seed_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    _save_json(output_dir / "best_seed.json", best_run)
    print(
        f"Best seed: {best_run['seed']} ({score_key}={best_run['score']:.6f}). "
        f"Promoted checkpoint to {output_dir / 'best.pth'}"
    )
    return best_run


def summarize_seed_results(seed_results, output_dir, task):
    best_run = promote_best_seed(seed_results, output_dir, task)
    summary = {"runs": seed_results, "best_run": best_run, "aggregate": {}}
    skip_keys = {"epoch", "lr", "train_loss"}
    metric_keys = sorted({
        key
        for run in seed_results
        for key, value in run.get("metrics", {}).items()
        if isinstance(value, (int, float)) and key not in skip_keys and value is not None
    })
    for key in metric_keys:
        values = [float(run["metrics"][key]) for run in seed_results if key in run.get("metrics", {}) and run["metrics"][key] is not None]
        if values:
            summary["aggregate"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    _save_json(Path(output_dir) / "multi_seed_summary.json", summary)
    return summary


class ASLSingleLabel(torch.nn.Module):
    """Single-label Asymmetric Loss (Ben-Baruch et al., 2021) for softmax outputs.

    Down-weights easy negatives (the majority class) via gamma_neg, addressing
    class imbalance the way the Vietnamese CXR study of Nguyen et al. (2022) did
    with ASL. gamma_neg > gamma_pos focuses learning on the minority positives.
    """

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, eps=0.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, inputs, target):
        num_classes = inputs.size(-1)
        log_preds = self.logsoftmax(inputs)
        targets = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        asymmetric_w = torch.pow(
            1 - xs_pos * targets - xs_neg * anti_targets,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w
        if self.eps > 0:
            targets = targets.mul(1 - self.eps).add(self.eps / num_classes)
        return -(targets * log_preds).sum(dim=-1).mean()


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


def run_experiment(args):
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
    if args.task in ("covid", "covid_qu_ex"):
        if args.loss == "asl":
            criterion = ASLSingleLabel(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos)
            print(f"Using ASLSingleLabel (gamma_neg={args.asl_gamma_neg}, gamma_pos={args.asl_gamma_pos})")
        elif args.class_weight == "balanced":
            # inverse-frequency (sklearn "balanced"): w_c = N / (K * count_c), read from train list
            counts = [0] * num_classes
            with open(args.covid_train_list, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    c = int(line.split()[-1])
                    if 0 <= c < num_classes:
                        counts[c] += 1
            total = sum(counts)
            w = torch.tensor([(total / (num_classes * n)) if n else 0.0 for n in counts], dtype=torch.float)
            criterion = torch.nn.CrossEntropyLoss(weight=w)
            print(f"Using class-balanced CrossEntropyLoss: counts={counts}, weights={w.tolist()}")
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
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
        export_requested_onnx(model, args, output_dir, device, eval_only=True)
        run_inference_benchmark(
            model, test_loader, args.task, device, label_names, args, output_dir,
            eval_only=True, pytorch_result=result,
        )
        run_grad_cam(model, test_loader, args.task, device, label_names, args, output_dir, eval_only=True)
        return result

    total_steps = len(train_loader) * args.epochs
    if start_epoch >= args.epochs:
        print(f"Nothing to train: resume epoch {start_epoch} >= --epochs {args.epochs}")
        return None
    best_result = None
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
            best_result = dict(result)
            _save_json(output_dir / "best_metrics.json", result)
        save_training_checkpoint(output_dir / "checkpoint.pth", model, optimizer, epoch, best_score, args, label_names)

    export_requested_onnx(model, args, output_dir, device)
    run_inference_benchmark(model, test_loader, args.task, device, label_names, args, output_dir)
    run_grad_cam(model, test_loader, args.task, device, label_names, args, output_dir)
    return best_result


def main():
    args = parse_args()
    if not args.seeds:
        run_experiment(args)
        return

    if args.resume_path:
        raise ValueError("--seeds cannot be combined with --resume_path; run each resumed seed separately.")
    if args.onnx_path or args.benchmark_output or args.grad_cam_dir:
        raise ValueError("When using --seeds, leave --onnx_path, --benchmark_output and --grad_cam_dir unset so each seed writes inside its own output folder.")

    seeds = parse_seed_list(args.seeds)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_post_after_seeds = args.export_onnx or args.compare_onnx or args.benchmark_inference or args.grad_cam
    seed_results = []
    for seed in seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.seeds = None
        run_args.seed = seed
        run_args.output_dir = str(base_output_dir / f"seed_{seed}")
        run_args.export_onnx = False
        run_args.compare_onnx = False
        run_args.benchmark_inference = False
        run_args.grad_cam = False
        print(f"=== Running seed {seed} -> {run_args.output_dir} ===")
        metrics_payload = run_experiment(run_args)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if metrics_payload is not None:
            seed_results.append({
                "seed": seed,
                "output_dir": run_args.output_dir,
                "metrics": metrics_payload,
            })

    summary = summarize_seed_results(seed_results, base_output_dir, args.task)
    print("Multi-seed summary:", json.dumps(summary["aggregate"]))

    best_run = summary.get("best_run")
    if run_post_after_seeds and best_run is not None:
        post_args = argparse.Namespace(**vars(args))
        post_args.seeds = None
        post_args.seed = int(best_run["seed"])
        post_args.output_dir = str(base_output_dir)
        post_args.checkpoint_path = str(base_output_dir / "best.pth")
        post_args.resume_path = None
        post_args.pre_train = False
        post_args.eval_only = True
        print(f"=== Running post-processing once with best seed {post_args.seed} ===")
        run_experiment(post_args)


if __name__ == "__main__":
    main()
