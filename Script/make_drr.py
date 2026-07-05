"""Memory-safe CT -> DRR generator using DiffDRR (one study at a time).

Processes each CT ZIP independently: read the main axial series, build one 3D
volume, render a frontal DRR on GPU, save a 512x512 PNG, then release all
memory before the next study. Peak RAM is bounded by a single CT volume, so it
cannot exhaust memory the way the batch exporter did.

DiffDRR (eigenvivek/DiffDRR) is pure-PyTorch, so it runs on the Blackwell GPU
where the official pycuda_drr kernel (texture references, removed in CUDA 12+)
cannot compile.

Usage:
    python make_drr.py --ct-dir Script/ANONYMIZE/CT \
        --out-dir Script/UniMiSSPlus_data/2D_images_drr \
        --sample-dir Script/results/drr_samples --limit 3
"""
import argparse
import gc
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import psutil
import pydicom


def ram_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def read_ct_series_from_zip(zip_path: Path):
    """Return (volume[Z,Y,X] float32 HU, spacing(z,y,x)) for the main axial series, or None."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
        if not names:
            return None
        # Group by series, read headers only (cheap).
        series = {}
        for n in names:
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(n)), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if str(ds.get("Modality", "")).upper() != "CT":
                continue
            uid = str(ds.get("SeriesInstanceUID", "none"))
            series.setdefault(uid, []).append(n)
        if not series:
            return None
        # Pick the series with the most slices (scouts/localizers have few) and >= 24.
        best_uid = max(series, key=lambda u: len(series[u]))
        slice_names = series[best_uid]
        if len(slice_names) < 24:
            return None

        # Read pixel data for the chosen series only.
        slices = []
        for n in slice_names:
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(n)), force=True)
                if not hasattr(ds, "pixel_array"):
                    continue
                pos = ds.get("ImagePositionPatient", [0, 0, 0])
                z = float(pos[2]) if pos else float(ds.get("InstanceNumber", 0))
                slope = float(ds.get("RescaleSlope", 1.0))
                intercept = float(ds.get("RescaleIntercept", 0.0))
                arr = ds.pixel_array.astype(np.float32) * slope + intercept  # HU
                ps = ds.get("PixelSpacing", [1.0, 1.0])
                thick = float(ds.get("SliceThickness", 1.0))
                slices.append((z, arr, float(ps[0]), float(ps[1]), thick))
            except Exception:
                continue
        if len(slices) < 24:
            return None
        slices.sort(key=lambda s: s[0])
        volume = np.stack([s[1] for s in slices], axis=0)  # [Z, Y, X]
        # z-spacing from actual slice positions if possible, else SliceThickness.
        zs = [s[0] for s in slices]
        dz = float(np.median(np.diff(zs))) if len(zs) > 1 and np.any(np.diff(zs)) else slices[0][4]
        spacing = (abs(dz) or 1.0, slices[0][2], slices[0][3])
        return volume.astype(np.float32), spacing


def crop_to_body(volume: np.ndarray, spacing, hu_thresh: float = -500.0, margin_mm: float = 15.0):
    """Crop the CT volume to the body (tissue) bounding box before projecting.

    Root cause of the black border: a chest CT's field of view is a square/circle
    containing the patient plus a large amount of surrounding air (often >50% of
    voxels at HU < -500). Projecting the full volume renders that air as dark
    margins. Cropping the volume to the tissue bbox here makes the body fill the
    rendered frame natively -- this is done at the volume level, not as a post-hoc
    crop of the rendered 2D image.
    """
    mask = volume > hu_thresh
    if not mask.any():
        return volume, spacing
    out = volume
    sl = []
    for axis in range(3):
        idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        m = int(round(margin_mm / max(spacing[axis], 1e-3)))
        lo = max(idx[0] - m, 0)
        hi = min(idx[-1] + m + 1, volume.shape[axis])
        sl.append(slice(lo, hi))
    return np.ascontiguousarray(out[tuple(sl)]), spacing


def downsample(volume: np.ndarray, spacing, max_dim: int):
    """Stride-downsample so every axis <= max_dim; scale spacing to match.

    Keeps GPU memory bounded: DiffDRR's Siddon renderer allocates tensors
    proportional to volume voxels x rays, so a full 512x512x300 CT at render
    height 512 needs ~14 GB VRAM. Downsampling the volume is the cheapest fix.
    """
    factors = [max(1, int(np.ceil(d / max_dim))) for d in volume.shape]
    if all(f == 1 for f in factors):
        return volume, spacing
    sl = tuple(slice(None, None, f) for f in factors)
    vol = volume[sl]
    new_spacing = tuple(spacing[i] * factors[i] for i in range(3))
    return np.ascontiguousarray(vol), new_spacing


def render_drr(volume: np.ndarray, spacing, device: str, max_dim: int = 448, height: int = 512):
    """volume[Z,Y,X] HU -> DRR 512x512 uint8 (dense = bright, radiograph look)."""
    import torch
    import nibabel as nib
    from PIL import Image
    from diffdrr.drr import DRR
    from diffdrr.data import read

    volume, spacing = crop_to_body(volume, spacing)   # remove surrounding air -> no dark border
    volume, spacing = downsample(volume, spacing, max_dim)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
        tmp = tf.name
    try:
        aff = np.eye(4)
        aff[0, 0], aff[1, 1], aff[2, 2] = spacing[2], spacing[1], spacing[0]
        # DiffDRR/torchio expect [X,Y,Z]-ordered array with matching affine; transpose Z,Y,X -> X,Y,Z.
        nib.save(nib.Nifti1Image(np.transpose(volume, (2, 1, 0)), aff), tmp)
        subject = read(tmp)
        # Match the detector field of view to the body's physical size so the
        # anatomy fills the frame with no dark margin. delx (detector pixel
        # spacing) = body_extent_mm / height; the small factor leaves a thin
        # margin so ribs/shoulders are not clipped. This is the geometric fix
        # for the border, replacing any post-hoc image crop.
        body_mm = max(volume.shape[1] * spacing[1], volume.shape[2] * spacing[2])
        delx = body_mm * 1.05 / height
        # Siddon exact ray-tracing (DiffDRR default): the exact line integral of
        # attenuation through the voxel grid. Trilinear interpolation is an
        # approximation that only converges to this at large n_points; it looks
        # smoother but is not "more realistic", so we use the exact method.
        drr = DRR(subject, sdd=1020.0, height=height, delx=float(delx)).to(device)
        rot = torch.zeros(1, 3, device=device)
        tr = torch.tensor([[0.0, 900.0, 0.0]], device=device)
        with torch.no_grad():
            img = drr(rot, tr, parameterization="euler_angles", convention="ZXY")
        arr = img[0, 0].detach().cpu().numpy().astype(np.float32)
        del drr, img, subject, rot, tr
        torch.cuda.empty_cache()
    finally:
        Path(tmp).unlink(missing_ok=True)

    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)  # dense (high attenuation) = bright
    u8 = (arr * 255).astype(np.uint8)
    return np.array(Image.fromarray(u8).resize((512, 512), Image.BILINEAR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", required=True, help="Folder of CT ZIPs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-dir")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N studies (0 = all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from PIL import Image

    ct_dir = Path(args.ct_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    if sample_dir:
        sample_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(ct_dir.glob("*.zip"))
    if args.limit:
        zips = zips[: args.limit]
    print(f"Processing {len(zips)} CT studies (one at a time). Start RAM: {ram_gb():.2f} GB")

    made, skipped = 0, 0
    for i, z in enumerate(zips, 1):
        try:
            result = read_ct_series_from_zip(z)
            if result is None:
                skipped += 1
                print(f"  [{i}/{len(zips)}] {z.stem[:24]}: skip (no usable >=24-slice CT series)")
                continue
            volume, spacing = result
            drr = render_drr(volume, spacing, args.device)
            out_path = out_dir / f"{z.stem}.png"
            Image.fromarray(drr).save(out_path)
            made += 1
            if sample_dir and made <= 12:
                Image.fromarray(drr).save(sample_dir / f"sample_{made:02d}_{z.stem}.png")
            print(f"  [{i}/{len(zips)}] {z.stem[:24]}: DRR ok  vol={volume.shape}  RAM={ram_gb():.2f} GB")
            del volume, drr, result
        except Exception as e:
            skipped += 1
            print(f"  [{i}/{len(zips)}] {z.stem[:24]}: ERROR {e!r}")
        finally:
            gc.collect()

    print(f"\nDone. Generated {made} DRR PNGs, skipped {skipped}. End RAM: {ram_gb():.2f} GB")
    print(f"Output: {out_dir}")
    if sample_dir:
        print(f"Samples: {sample_dir}")


if __name__ == "__main__":
    main()
