"""
Benchmark inference speed of MiT+ encoder/full model.
Measures: latency (ms), throughput (samples/sec), GPU memory usage, parameter count.

Usage:
  python benchmark_speed.py                    # run all benchmarks
  python benchmark_speed.py --mode 2D          # encoder 2D only
  python benchmark_speed.py --mode 3D          # encoder 3D only
  python benchmark_speed.py --mode full        # full encoder+decoder
  python benchmark_speed.py --device cpu       # force CPU
  python benchmark_speed.py --batch_size 4     # change batch size
"""

import argparse
import time
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from MiTplus import MiTplus_encoder, MiTplus


# ── helpers ──────────────────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def gpu_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 ** 2
    return 0.0


def warmup(model, dummy, modal_type, device, n=10):
    with torch.no_grad():
        for _ in range(n):
            _ = model(dummy, modal_type) if modal_type else model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, dummy, modal_type, device, n_runs=100):
    """Returns mean latency (ms) and std over n_runs forward passes."""
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(dummy, modal_type) if modal_type else model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return arr.mean(), arr.std(), arr.min(), arr.max()


def print_results(tag, batch_size, mean_ms, std_ms, min_ms, max_ms, params, mem_mb, device):
    throughput = 1000.0 / mean_ms * batch_size
    print(f"\n{'=' * 58}")
    print(f"  {tag}")
    print(f"{'=' * 58}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Params total : {params[0]/1e6:.2f} M  |  trainable: {params[1]/1e6:.2f} M")
    print(f"  Latency      : {mean_ms:.2f} ± {std_ms:.2f} ms  (min {min_ms:.2f} / max {max_ms:.2f})")
    print(f"  Throughput   : {throughput:.1f} samples/sec")
    if mem_mb > 0:
        print(f"  GPU memory   : {mem_mb:.1f} MB")


# ── benchmark functions ───────────────────────────────────────────────────────

def bench_encoder_2D(device, batch_size, n_runs, img_size=224):
    print(f"\n[INFO] Building encoder (2D mode), img_size={img_size} ...")
    model = MiTplus_encoder(modal_type='2D', img_size2D=img_size).to(device).eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    warmup(model, dummy, '2D', device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mean_ms, std_ms, min_ms, max_ms = benchmark(model, dummy, '2D', device, n_runs)
    mem = gpu_memory_mb(device)
    print_results(
        f"Encoder-only | mode=2D | input={batch_size}×3×{img_size}×{img_size}",
        batch_size, mean_ms, std_ms, min_ms, max_ms, count_params(model), mem, device
    )


def bench_encoder_3D(device, batch_size, n_runs, img_size3D=None):
    if img_size3D is None:
        img_size3D = [16, 96, 96]
    D, H, W = img_size3D
    print(f"\n[INFO] Building encoder (3D mode), img_size3D={img_size3D} ...")
    model = MiTplus_encoder(modal_type='3D', img_size3D=img_size3D).to(device).eval()
    dummy = torch.randn(batch_size, 1, D, H, W, device=device)

    warmup(model, dummy, '3D', device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mean_ms, std_ms, min_ms, max_ms = benchmark(model, dummy, '3D', device, n_runs)
    mem = gpu_memory_mb(device)
    print_results(
        f"Encoder-only | mode=3D | input={batch_size}×1×{D}×{H}×{W}",
        batch_size, mean_ms, std_ms, min_ms, max_ms, count_params(model), mem, device
    )


def bench_full_2D(device, batch_size, n_runs, img_size=224):
    print(f"\n[INFO] Building full model (encoder+decoder, 2D mode), img_size={img_size} ...")
    model = MiTplus(modal_type='2D', img_size2D=img_size).to(device).eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # full model uses forward(inputs) without modal_type arg
    warmup(model, dummy, None, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mean_ms, std_ms, min_ms, max_ms = benchmark(model, dummy, None, device, n_runs)
    mem = gpu_memory_mb(device)
    print_results(
        f"Full model   | mode=2D | input={batch_size}×3×{img_size}×{img_size}",
        batch_size, mean_ms, std_ms, min_ms, max_ms, count_params(model), mem, device
    )


def bench_full_3D(device, batch_size, n_runs, img_size3D=None):
    if img_size3D is None:
        img_size3D = [16, 96, 96]
    D, H, W = img_size3D
    print(f"\n[INFO] Building full model (encoder+decoder, 3D mode), img_size3D={img_size3D} ...")
    model = MiTplus(modal_type='3D', img_size3D=img_size3D).to(device).eval()
    dummy = torch.randn(batch_size, 1, D, H, W, device=device)

    warmup(model, dummy, None, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mean_ms, std_ms, min_ms, max_ms = benchmark(model, dummy, None, device, n_runs)
    mem = gpu_memory_mb(device)
    print_results(
        f"Full model   | mode=3D | input={batch_size}×1×{D}×{H}×{W}",
        batch_size, mean_ms, std_ms, min_ms, max_ms, count_params(model), mem, device
    )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MiT+ inference speed benchmark")
    p.add_argument("--mode", choices=["2D", "3D", "full", "all"], default="all",
                   help="Which benchmark to run (default: all)")
    p.add_argument("--device", default="auto",
                   help="cuda / cpu / auto (default: auto → uses GPU if available)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size for benchmark (default: 1)")
    p.add_argument("--n_runs", type=int, default=100,
                   help="Number of forward passes to average over (default: 100)")
    p.add_argument("--img_size2D", type=int, default=224,
                   help="2D image size (default: 224)")
    p.add_argument("--img_size3D", type=int, nargs=3, default=[16, 96, 96],
                   metavar=("D", "H", "W"),
                   help="3D volume size D H W (default: 16 96 96)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nPyTorch version : {torch.__version__}")
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(device)}")

    run_2D  = args.mode in ("2D",  "all")
    run_3D  = args.mode in ("3D",  "all")
    run_full = args.mode in ("full", "all")

    try:
        if run_2D:
            bench_encoder_2D(device, args.batch_size, args.n_runs, args.img_size2D)

        if run_3D:
            bench_encoder_3D(device, args.batch_size, args.n_runs, args.img_size3D)

        if run_full:
            bench_full_2D(device, args.batch_size, args.n_runs, args.img_size2D)
            bench_full_3D(device, args.batch_size, args.n_runs, args.img_size3D)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

    print("\n[Done]")


if __name__ == "__main__":
    main()
