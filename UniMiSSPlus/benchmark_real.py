"""
Benchmark inference speed using REAL sample data from the project.

- 3D: loads .nii.gz subvolumes, applies same preprocessing as data_loader3D2D.py
      crop_size_3D=(64, 256, 256), normalize /1024, pad to 256×256
- 2D: loads .png X-ray images (512×512 RGB), resize to 256×256

Usage:
  python benchmark_real.py
  python benchmark_real.py --n_runs 50 --batch_size 2
"""

import argparse
import time
import os
import sys
import math

import numpy as np
import torch
import nibabel as nib
import cv2
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from MiTplus import MiTplus_encoder


DATA_3D = os.path.join(os.path.dirname(__file__), "data", "3D_subvolumes_examples")
DATA_2D = os.path.join(os.path.dirname(__file__), "data", "2D_images_examples")

CROP_D, CROP_H, CROP_W = 64, 256, 256
IMG_SIZE_2D = 256


# ── preprocessing (mirrors data_loader3D2D.py) ───────────────────────────────

def pad_image(img, crop_d):
    """Pad H, W to 256; pad D to crop_d."""
    rows_missing = max(0, math.ceil(256 - img.shape[0]))
    cols_missing = max(0, math.ceil(256 - img.shape[1]))
    dept_missing = max(0, math.ceil(crop_d - img.shape[2]))
    return np.pad(img,
                  ((0, rows_missing), (0, cols_missing),
                   (dept_missing // 2, dept_missing - dept_missing // 2)),
                  'constant')


def load_3d_volume(path):
    """Load one .nii.gz, pad, normalize, return float32 tensor (1, D, H, W)."""
    arr = nib.load(path).get_fdata()          # (H, W, D) raw
    arr = pad_image(arr, CROP_D)              # ensure >= 256×256×CROP_D
    arr = arr[np.newaxis]                     # (1, H, W, D)
    arr = arr.transpose(0, 3, 1, 2)           # (1, D, H, W)

    # centre-crop to exact (CROP_D, CROP_H, CROP_W)
    _, d, h, w = arr.shape
    d0 = (d - CROP_D) // 2
    h0 = (h - CROP_H) // 2
    w0 = (w - CROP_W) // 2
    arr = arr[:, d0:d0+CROP_D, h0:h0+CROP_H, w0:w0+CROP_W]

    arr = arr.astype(np.float32) / 1024.0     # normalize as in truncate()
    return torch.from_numpy(arr)              # (1, 64, 256, 256)


def load_2d_image(path):
    """Load one .png, convert BGR→RGB, resize to IMG_SIZE_2D, return float32 tensor (3, H, W)."""
    img = cv2.imread(path)                    # (512, 512, 3) BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE_2D, IMG_SIZE_2D))
    t = transforms.ToTensor()(img)            # (3, 256, 256), float [0,1]
    return t


# ── build batches from real files ────────────────────────────────────────────

def build_batch_3d(batch_size):
    files = sorted([os.path.join(DATA_3D, f)
                    for f in os.listdir(DATA_3D) if f.endswith('.nii.gz')])
    vols = [load_3d_volume(f) for f in files[:batch_size]]
    # repeat if batch_size > available files
    while len(vols) < batch_size:
        vols.extend(vols)
    vols = vols[:batch_size]
    return torch.stack(vols)                  # (B, 1, 64, 256, 256)


def build_batch_2d(batch_size):
    files = sorted([os.path.join(DATA_2D, f)
                    for f in os.listdir(DATA_2D) if f.endswith('.png')])
    imgs = [load_2d_image(f) for f in files[:batch_size]]
    while len(imgs) < batch_size:
        imgs.extend(imgs)
    imgs = imgs[:batch_size]
    return torch.stack(imgs)                  # (B, 3, 256, 256)


# ── benchmark helpers ─────────────────────────────────────────────────────────

def warmup(model, x, modal_type, device, n=10):
    with torch.no_grad():
        for _ in range(n):
            model(x, modal_type)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(model, x, modal_type, device, n_runs):
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x, modal_type)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), arr.std(), arr.min(), arr.max()


def report(tag, x_shape, batch_size, mean_ms, std_ms, min_ms, max_ms, params, mem_mb, device):
    throughput = 1000.0 / mean_ms * batch_size
    print(f"\n{'=' * 62}")
    print(f"  {tag}")
    print(f"{'=' * 62}")
    print(f"  Device        : {device}")
    print(f"  Input shape   : {tuple(x_shape)}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Params        : {params/1e6:.2f} M")
    print(f"  Latency       : {mean_ms:.2f} ± {std_ms:.2f} ms  (min {min_ms:.2f} / max {max_ms:.2f})")
    print(f"  Throughput    : {throughput:.1f} samples/sec")
    if mem_mb > 0:
        print(f"  GPU memory    : {mem_mb:.1f} MB")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    print(f"\nPyTorch : {torch.__version__}")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(device)}")

    # ── load real data ────────────────────────────────────────────────────────
    print("\n[INFO] Loading real 3D samples ...")
    batch_3d = build_batch_3d(args.batch_size).to(device)
    print(f"       3D batch shape : {tuple(batch_3d.shape)}  "
          f"value range [{batch_3d.min():.3f}, {batch_3d.max():.3f}]")

    print("[INFO] Loading real 2D samples ...")
    batch_2d = build_batch_2d(args.batch_size).to(device)
    print(f"       2D batch shape : {tuple(batch_2d.shape)}  "
          f"value range [{batch_2d.min():.3f}, {batch_2d.max():.3f}]")

    # ── benchmark encoder 2D ─────────────────────────────────────────────────
    print("\n[INFO] Building encoder (2D mode) ...")
    enc2d = MiTplus_encoder(modal_type='2D', img_size2D=IMG_SIZE_2D).to(device).eval()
    params2d = sum(p.numel() for p in enc2d.parameters())

    warmup(enc2d, batch_2d, '2D', device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    m, s, mn, mx = run_benchmark(enc2d, batch_2d, '2D', device, args.n_runs)
    mem2d = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0
    report("Encoder-only | mode=2D | REAL X-ray data",
           batch_2d.shape, args.batch_size, m, s, mn, mx, params2d, mem2d, device)

    # ── benchmark encoder 3D ─────────────────────────────────────────────────
    print("\n[INFO] Building encoder (3D mode) ...")
    enc3d = MiTplus_encoder(modal_type='3D',
                            img_size3D=[CROP_D, CROP_H, CROP_W]).to(device).eval()
    params3d = sum(p.numel() for p in enc3d.parameters())

    warmup(enc3d, batch_3d, '3D', device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    m, s, mn, mx = run_benchmark(enc3d, batch_3d, '3D', device, args.n_runs)
    mem3d = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0
    report("Encoder-only | mode=3D | REAL CT subvolume data",
           batch_3d.shape, args.batch_size, m, s, mn, mx, params3d, mem3d, device)

    print("\n[Done]")


if __name__ == "__main__":
    main()
