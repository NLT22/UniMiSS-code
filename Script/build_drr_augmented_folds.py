"""Add CT-DRR images to each CV fold's TRAIN set, patient-safe.

For every fold, appends eligible CT-DRR PNGs to train_oversampled.txt. A DRR is
eligible only if its source patient is NOT in that fold's val or test split, so
no patient leaks across CT-DRR (train) and real X-ray (val/test). DRRs never go
into val/test -- evaluation stays on real X-rays only.

Join key is the CT ZIP stem: DRR png stem == zip stem == patient_study_map
study_id == Path(zip_path column in labels.xlsx).stem.

Usage:
    python build_drr_augmented_folds.py \
        --cv-dir Script/labels/vietnam_xray_cv \
        --drr-dir Script/UniMiSSPlus_data/2D_images_drr \
        --labels Script/labels.xlsx \
        --patient-map Script/ANONYMIZE_meta/patient_study_map.csv \
        --out-dir Script/labels/vietnam_xray_cv_drr \
        --abnormal-repeat 3
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    if str(path).lower().endswith(".xlsx"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active; it = ws.iter_rows(values_only=True)
        hdr = [str(h) if h is not None else "" for h in next(it, [])]
        rows = [{hdr[j]: ("" if v is None else str(v)) for j, v in enumerate(vals) if j < len(hdr)}
                for vals in it if vals and not all(v is None for v in vals)]
        wb.close(); return rows
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", required=True)
    ap.add_argument("--drr-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--patient-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--abnormal-repeat", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cv_dir = Path(args.cv_dir)
    drr_dir = Path(args.drr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # stem -> label class ("0"=Abnormal, "1"=Normal), CT rows only
    ct_label = {}
    for r in read_csv(args.labels):
        if r.get("modality") != "CT":
            continue
        zp = r.get("zip_path", "")
        stem = Path(zp).stem if zp else ""
        cls = r.get("normal_abnormal_class", "")
        if stem and cls in ("0", "1"):
            ct_label[stem] = cls

    # stem -> patient_hash
    stem_patient = {}
    for r in read_csv(args.patient_map):
        if r.get("modality") == "CT" and r.get("patient_hash"):
            stem_patient[r["study_id"]] = r["patient_hash"]

    # DRRs that actually exist
    drr_stems = [p.stem for p in drr_dir.glob("*.png")]
    drr_info = []  # (stem, class, patient)
    missing_label = missing_patient = 0
    for stem in drr_stems:
        cls = ct_label.get(stem)
        pat = stem_patient.get(stem)
        if cls is None:
            missing_label += 1
            continue
        if pat is None:
            missing_patient += 1
            continue
        drr_info.append((stem, cls, pat))

    print(f"DRR PNGs found: {len(drr_stems)} | with label+patient: {len(drr_info)} "
          f"(missing label {missing_label}, missing patient {missing_patient})")
    n_abn = sum(1 for _, c, _ in drr_info if c == "0")
    print(f"  DRR Abnormal: {n_abn}  Normal: {len(drr_info) - n_abn}")

    # per-fold val+test patient sets from the manifest
    manifest = read_csv(cv_dir / "cv_manifest.csv")
    heldout_patients = defaultdict(set)  # fold -> {patient_hash in val or test}
    for r in manifest:
        if r["split"] in ("val", "test"):
            heldout_patients[r["fold"]].add(r.get("patient_hash", ""))

    summary = []
    for fold in range(args.folds):
        src_fold = cv_dir / f"fold_{fold}"
        dst_fold = out_dir / f"fold_{fold}"
        dst_fold.mkdir(parents=True, exist_ok=True)

        # copy val/test/train verbatim (evaluation unchanged, on real X-rays)
        for fname in ("val.txt", "test.txt", "train.txt"):
            shutil.copy(src_fold / fname, dst_fold / fname)

        held = heldout_patients[str(fold)]
        eligible = [(s, c) for s, c, p in drr_info if p not in held]
        added_lines = []
        added_abn = 0
        for stem, cls in eligible:
            reps = args.abnormal_repeat if cls == "0" else 1
            added_lines.extend([f"2D_images_drr/{stem}.png {cls}"] * reps)
            if cls == "0":
                added_abn += 1

        # augmented train = original oversampled X-ray train + eligible DRRs
        base = (src_fold / "train_oversampled.txt").read_text(encoding="utf-8").splitlines()
        aug = [l for l in base if l.strip()] + added_lines
        (dst_fold / "train_oversampled.txt").write_text("\n".join(aug) + "\n", encoding="utf-8")

        excluded = len(drr_info) - len(eligible)
        summary.append((fold, len(base), len(added_lines), added_abn, excluded))
        print(f"fold {fold}: X-ray train {len(base)} + DRR rows {len(added_lines)} "
              f"({added_abn} abn studies x{args.abnormal_repeat}); excluded {excluded} DRR (patient in val/test)")

    with (out_dir / "drr_augment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "xray_train_rows", "drr_rows_added", "drr_abnormal_studies", "drr_excluded_leakage"])
        w.writerows(summary)
    print(f"\nWrote augmented folds to {out_dir}")


if __name__ == "__main__":
    main()
