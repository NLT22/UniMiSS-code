import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


COVID_DEFAULT_CLASSES = ("COVID", "Lung_Opacity", "Normal")
COVID_OTHER_NORMAL_CLASSES = ("COVID", "Other", "Normal")
COVID_QU_EX_CLASSES = ("COVID-19", "Non-COVID", "Normal")
COVID_QU_EX_SPLIT_DIRS = {
    "train": ("Train", "Val"),
    "test": ("Test",),
}

NIH_DISEASE_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
NIH_LABELS = NIH_DISEASE_LABELS + ("No Finding",)


def build_transform_classification(normalize="chestx-ray", crop_size=224, resize=256, mode="train", test_augment=True):
    if normalize.lower() == "imagenet":
        normalize_op = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
        normalize_op = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
        normalize_op = None
    else:
        raise ValueError(f"Unknown normalization preset: {normalize}")

    ops = []
    if mode == "train":
        ops.extend([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
        ])
        if normalize_op is not None:
            ops.append(normalize_op)
    elif mode in ("valid", "test"):
        ops.append(transforms.Resize((resize, resize)))
        if mode == "test" and test_augment:
            ops.extend([
                transforms.TenCrop(crop_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            ])
            if normalize_op is not None:
                ops.append(transforms.Lambda(lambda crops: torch.stack([normalize_op(crop) for crop in crops])))
        else:
            ops.extend([transforms.CenterCrop(crop_size), transforms.ToTensor()])
            if normalize_op is not None:
                ops.append(normalize_op)
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    return transforms.Compose(ops)


def _read_name_list(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_labeled_list(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(f"Expected '<image_path> <label>' in {path}, got: {line}")
            samples.append((parts[0], int(parts[1])))
    return samples


def _image_files(path):
    path = Path(path)
    files = []
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(path.glob(suffix))
    return sorted([p for p in files if p.is_file()])


def build_covid_samples(root, classes=COVID_DEFAULT_CLASSES, train_list=None, test_list=None, split="train",
                        test_split=0.2, seed=1234):
    root = Path(root)
    if split == "train" and train_list:
        return [(root / rel_path, label) for rel_path, label in _parse_labeled_list(train_list)]
    if split != "train" and test_list:
        return [(root / rel_path, label) for rel_path, label in _parse_labeled_list(test_list)]

    rng = random.Random(seed)
    selected = []
    for label, class_name in enumerate(classes):
        image_dir = root / class_name / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"COVID class image folder not found: {image_dir}")
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        rng.shuffle(image_paths)
        n_test = int(round(len(image_paths) * test_split))
        if split == "train":
            class_paths = image_paths[n_test:]
        else:
            class_paths = image_paths[:n_test]
        selected.extend((path, label) for path in class_paths)

    rng.shuffle(selected)
    return selected


def build_covid_other_normal_samples(root, train_list=None, test_list=None, split="train", test_split=0.2, seed=1234):
    root = Path(root)
    if split == "train" and train_list:
        return [(root / rel_path, label) for rel_path, label in _parse_labeled_list(train_list)]
    if split != "train" and test_list:
        return [(root / rel_path, label) for rel_path, label in _parse_labeled_list(test_list)]

    groups = (
        ("COVID", ("COVID",)),
        ("Other", ("Lung_Opacity", "Viral Pneumonia")),
        ("Normal", ("Normal",)),
    )
    rng = random.Random(seed)
    selected = []
    for label, (_, class_names) in enumerate(groups):
        group_paths = []
        for class_name in class_names:
            image_dir = root / class_name / "images"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"COVID class image folder not found: {image_dir}")
            group_paths.extend(_image_files(image_dir))
        rng.shuffle(group_paths)
        n_test = int(round(len(group_paths) * test_split))
        if split == "train":
            class_paths = group_paths[n_test:]
        else:
            class_paths = group_paths[:n_test]
        selected.extend((path, label) for path in class_paths)

    rng.shuffle(selected)
    return selected


def _resolve_covid_qu_ex_base(root, subset="lung"):
    root = Path(root)
    subset_names = {
        "lung": "Lung Segmentation Data",
        "infection": "Infection Segmentation Data",
    }
    preferred = subset_names[subset]
    candidates = [
        root,
        root / preferred / preferred,
        root / "Lung Segmentation Data" / "Lung Segmentation Data",
        root / "Infection Segmentation Data" / "Infection Segmentation Data",
    ]
    for candidate in candidates:
        if all((candidate / split_name).is_dir() for split_name in ("Train", "Val", "Test")):
            return candidate
    raise FileNotFoundError(
        f"Could not find COVID-QU-Ex split folders Train/Val/Test under {root}. "
        "Pass the dataset root or the nested '.../Lung Segmentation Data/Lung Segmentation Data' folder."
    )


def build_covid_qu_ex_samples(root, split="train", subset="lung", train_splits=("Train", "Val"),
                              test_splits=("Test",)):
    base = _resolve_covid_qu_ex_base(root, subset)
    split_names = train_splits if split == "train" else test_splits
    samples = []
    for split_name in split_names:
        for label, class_name in enumerate(COVID_QU_EX_CLASSES):
            image_dir = base / split_name / class_name / "images"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"COVID-QU-Ex image folder not found: {image_dir}")
            samples.extend((path, label) for path in _image_files(image_dir))
    return samples


def _index_nih_images(root):
    root = Path(root)
    image_map = {}
    for path in root.glob("images_*/images/*"):
        if path.is_file():
            image_map[path.name] = path
    if not image_map:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                image_map[path.name] = path
    return image_map


def _read_nih_csv(csv_path):
    labels_by_image = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["Image Index"]
            finding_labels = row["Finding Labels"].split("|")
            target = np.zeros(len(NIH_LABELS), dtype=np.float32)
            if "No Finding" in finding_labels:
                target[NIH_LABELS.index("No Finding")] = 1.0
            else:
                for label in finding_labels:
                    if label in NIH_DISEASE_LABELS:
                        target[NIH_LABELS.index(label)] = 1.0
            labels_by_image[image_name] = target
    return labels_by_image


def build_nih_samples(root, csv_path, list_path):
    image_names = _read_name_list(list_path)
    image_map = _index_nih_images(root)
    labels_by_image = _read_nih_csv(csv_path)

    samples = []
    missing_images = []
    missing_labels = []
    for image_name in image_names:
        if image_name not in image_map:
            missing_images.append(image_name)
            continue
        if image_name not in labels_by_image:
            missing_labels.append(image_name)
            continue
        samples.append((image_map[image_name], labels_by_image[image_name]))

    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} NIH images from {list_path} were not found under {root}. First: {preview}")
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(f"{len(missing_labels)} NIH images from {list_path} have no labels in {csv_path}. First: {preview}")
    return samples


class CovidClassificationDataset(data.Dataset):
    def __init__(self, root, split="train", classes=COVID_DEFAULT_CLASSES, train_list=None, test_list=None,
                 test_split=0.2, seed=1234, crop_size=224, resize=256, normalize="chestx-ray",
                 test_augment=False):
        self.samples = build_covid_samples(root, classes, train_list, test_list, split, test_split, seed)
        self.classes = tuple(classes)
        mode = "train" if split == "train" else "test"
        self.transform = build_transform_classification(normalize, crop_size, resize, mode, test_augment)
        print(f"Loaded {len(self.samples)} COVID {split} images from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(label)


class CovidOtherNormalDataset(data.Dataset):
    def __init__(self, root, split="train", train_list=None, test_list=None, test_split=0.2, seed=1234,
                 crop_size=224, resize=256, normalize="chestx-ray", test_augment=False):
        self.samples = build_covid_other_normal_samples(root, train_list, test_list, split, test_split, seed)
        self.classes = COVID_OTHER_NORMAL_CLASSES
        mode = "train" if split == "train" else "test"
        self.transform = build_transform_classification(normalize, crop_size, resize, mode, test_augment)
        print(f"Loaded {len(self.samples)} COVID other-normal {split} images from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(label)


class CovidQUExClassificationDataset(data.Dataset):
    def __init__(self, root, split="train", subset="lung", train_splits=("Train", "Val"), test_splits=("Test",),
                 crop_size=224, resize=256, normalize="chestx-ray", test_augment=False):
        self.samples = build_covid_qu_ex_samples(root, split, subset, train_splits, test_splits)
        self.classes = COVID_QU_EX_CLASSES
        mode = "train" if split == "train" else "test"
        self.transform = build_transform_classification(normalize, crop_size, resize, mode, test_augment)
        print(f"Loaded {len(self.samples)} COVID-QU-Ex {split} images from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(label)


class NIHChestXrayDataset(data.Dataset):
    def __init__(self, root, csv_path, list_path, split="train", crop_size=224, resize=256,
                 normalize="chestx-ray", test_augment=False):
        self.samples = build_nih_samples(root, csv_path, list_path)
        mode = "train" if split == "train" else "test"
        self.transform = build_transform_classification(normalize, crop_size, resize, mode, test_augment)
        print(f"Loaded {len(self.samples)} NIH {split} images from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), torch.from_numpy(target.copy())
