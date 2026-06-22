# UniMiSS DICOM Project Transfer Summary

This file summarizes the current project state for continuing in another Codex session.

## Project Goal

Prepare hospital DICOM chest CT/X-ray data for:

1. UniMiSSPlus-style upstream/self-supervised or domain-adaptation use.
2. Downstream X-ray-only classification.

Current downstream target:

```text
X-ray Normal vs Abnormal
```

Uncertain labels should be excluded from first training.

## Repository Context

Working folder:

```text
D:\2025.2\UniMiSS-code\Script
```

Important project folders:

```text
Script/        DICOM analyzer, anonymizer, exporter, labeler
UniMiSSPlus/   Frozen research code for UniMiSSPlus
docs/          Local papers
```

Main files:

```text
Script/dicom_analyzer.py
Script/dicom_labeler.py
Script/DICOM_README.md
Script/LABELS.lnk
```

`LABELS.lnk` points to:

```text
D:\OneDrive - Hanoi University of Science and Technology\LABELS.xlsx
```

On another computer, copy the actual `LABELS.xlsx`; the `.lnk` shortcut may break.

## DICOM Analyzer State

`dicom_analyzer.py` supports:

```text
analyze
anonymize
export-unimissplus
verify-unimissplus
```

Important behaviors:

- Filters to CHEST/THORAX only.
- Original files are never modified.
- ZIP input stays ZIP output.
- ZIP files inside folders stay ZIP files.
- CT scout/localizer series are skipped.
- CT series are grouped by `SeriesInstanceUID`.
- Different CT slice thickness/reconstruction series are not merged.
- 1.25 mm and 5 mm CT reconstructions are separate series.

## Anonymization Decisions

Preserve:

```text
PatientAge
PatientSex
```

Hash:

```text
PatientID
StudyID
AccessionNumber
StudyInstanceUID
SeriesInstanceUID
SOPInstanceUID
FrameOfReferenceUID
MediaStorageSOPInstanceUID
```

Remove or replace:

```text
PatientName -> ANONYMOUS
PatientBirthDate
PatientWeight
PatientAddress
OtherPatientIDs
OtherPatientNames
PatientComments
StudyDate / StudyTime
SeriesDate / SeriesTime
AcquisitionDate / AcquisitionTime
ContentDate / ContentTime
InstitutionAddress
InstitutionalDepartmentName
DeviceSerialNumber
StationName
ProtocolName
Physician/operator tags
```

Note:

```text
Pixel data with burned-in text is not modified.
```

## UniMiSSPlus Data Export

Command:

```powershell
python dicom_analyzer.py export-unimissplus <input> <output_data_dir>
```

Output structure:

```text
<output_data_dir>/
  2D_images/*.png
  3D_images/*.nii.gz
  3D_subvolumes/*.nii.gz
  2D_images.txt
  3D_images.txt
  DRR_REQUIRED.txt
```

Meaning:

```text
2D_images/*.png        real X-ray images exported from DICOM
3D_images/*.nii.gz     full CT volumes
3D_subvolumes/*.nii.gz CT chunks, default 24 slices
2D_images.txt          list of real X-ray PNGs
3D_images.txt          list of CT subvolume NIfTI files
```

Official DRR generation is still required before UniMiSSPlus upstream pretraining.

## What DRR Does

DRR means:

```text
Digitally Reconstructed Radiograph
```

It creates a synthetic 2D X-ray-like image from a 3D CT volume/subvolume.

In UniMiSSPlus:

```text
3D_subvolumes/example_dep0.nii.gz
-> DRR creates
3D_subvolumes/example_dep0.png
```

The loader expects both files:

```text
3D_subvolumes/*.nii.gz
3D_subvolumes/*.png
```

Official script:

```text
UniMiSSPlus/pycuda_drr/rendering_DL.py
```

Hardcoded paths in official code:

```text
input:  ../data/3D_subvolumes
output: ../data/3D_subvolumes
```

Run later in Linux/CUDA environment:

```bash
cd UniMiSSPlus/pycuda_drr
python setup.py install
python rendering_DL.py
```

## UniMiSSPlus Strategy

Use CT + X-ray for upstream/domain adaptation:

```text
CT + X-ray -> self-supervised representation learning
```

Use X-ray only for supervised downstream:

```text
X-ray -> Normal vs Abnormal classification
```

Avoid training a single supervised classifier on mixed CT + X-ray, because the model may learn modality shortcuts.

## Label File Rules

Source:

```text
LABELS.xlsx
```

Columns:

```text
Column 1: external study/archive ID
Column 2: doctor report/detail/conclusion text
Column 3: optional notes
```

User rule:

```text
Rows with empty detail column are default normal X-ray samples.
```

CT rule:

```text
Rows whose detail column contains "CT" are CT samples.
All others with nonempty detail are X-ray samples.
```

## Current Excel Counts

Based on direct Excel reading on 2026-06-19:

```text
Total rows: 778
Unique IDs: 776
Duplicate IDs: 2
```

Modality counts using user rule:

```text
CT: 98
X-ray: 680
```

Overall label counts:

```text
Normal: 446
Abnormal: 313
Uncertain: 19
```

By modality:

```text
CT:
  Normal: 5
  Abnormal: 90
  Uncertain: 3

X-ray:
  Normal: 441
  Abnormal: 223
  Uncertain: 16
```

For X-ray downstream excluding uncertain:

```text
Usable X-ray:
  Normal: 441
  Abnormal: 223
  Total: 664
```

Recommended balanced X-ray dataset:

```text
Train:
  156 Abnormal
  156 Normal

Validation:
  33 Abnormal
  33 Normal

Test:
  34 Abnormal
  34 Normal
```

This uses:

```text
223 Abnormal
223 Normal
```

Remaining normal reserve:

```text
218 Normal
```

## Labeler State

Main file:

```text
Script/dicom_labeler.py
```

Commands:

```powershell
python dicom_labeler.py extract LABELS.lnk DATA --output labels_raw_from_excel.csv
python dicom_labeler.py classify labels_raw_from_excel.csv --output labels_classified_from_excel.csv
python dicom_labeler.py build-lists labels_classified_from_excel.csv DATA <UniMiSSPlus_data_dir> --output-dir labels
```

Current classifier output fields:

```text
coarse_label
normal_abnormal_label
normal_abnormal_class
disease_label
multi_labels
confidence
needs_review
evidence
label_reason
conclusion
conclusion_full
```

Binary training labels:

```text
Abnormal = 0
Normal = 1
Uncertain = EXCLUDE
```

The labeler keeps `disease_label` only as secondary/review information.

## Normal Pattern

The following conclusion pattern should be normal:

```text
KẾT LUẬN:
--Hình ảnh X – Quang ngực thẳng không thấy bất thường.
```

After normalization:

```text
khong thay bat thuong
```

It maps to:

```text
coarse_label = NORMAL
normal_abnormal_label = Normal
normal_abnormal_class = 1
```

## Abnormal Patterns Added

Important abnormal patterns already added include:

```text
quai dong mach chu vong
cung dong mach chu vong
dong mach chu vong
quai dong mach chu
xo hoa
xo rai rac
xo gian phe quan
gian phe quan
dai xo
canh xo
day to chuc ke
day ke
dai mo
dam mo
mo ngoai vi
mo tuong doi thuan nhat
mo goc suon hoanh
goc suon hoanh ben trai tu
goc suon hoanh trai tu
day dinh mang phoi
dinh mang phoi
dan luu
dan luu khoang mang phoi
xep nhe
xep phoi
dap phoi
```

Important correction:

Do not use invented wording like:

```text
góc sườn hoành mờ
```

The actual observed wording was:

```text
mờ góc sườn hoành trái
Góc sườn hoành bên trái tù
```

Medical label wording must stay exact. Do not rewrite clinical phrases silently.

## X-ray Uncertain Rows

There were 16 X-ray uncertain rows before later rule additions. These should be reviewed manually or excluded.

Examples included:

```text
Hình ảnh X – Quang ngực thẳng bóng tim không to trường phổi hai bên sáng
Hình ảnh mờ nền phổi trái
Hình ảnh bóng tim to.
Hình ảnh X – Quang ngực thẳng bóng tim to, 2 phổi sáng.
Hình ảnh X – Quang ngực thẳng: Mờ thuần nhất góc sườn hoành trái.
Hình ảnh X – Quang ngực thẳng: Gãy 1/3 ngoài xương đòn trái.
Hình ảnh X – Quang ngực thẳng Bóng tim không to, quai ĐMC vồng, trường phổi hai bên sáng đều
Hình ảnh X – Quang ngực thẳng -- Trường phổi hai bên sáng đều
Hình ảnh X – Quang ngực thẳng bóng tim không to , quai động mạch vồng .
Hình ảnh X – Quang ngực thẳng: bóng tim to, hai trường phổi sáng. Góc sườn hoành phải kém nhọn.
Hình ảnh X – Quang ngực thẳng :- Bóng tim không to, hai phổi sáng.
Hình ảnh X – Quang ngực thẳng :Bóng tim to, trường phổi hai bên sáng.
Hình ảnh X – Quang ngực thẳng: Tăng các nhánh phế huyết hai phế trường.
Hình ảnh Dày các nhánh phế huyết quản hai phổi.
Hình ảnh X – Quang ngực thẳng: bóng tim không to, trường phổi 2 bên sáng.
Hình ảnh X – Quang ngực thẳng: mờ gần hoàn toàn trường phổi phải. Phổi trái sáng. Bóng tim không to
```

Some of these may become abnormal if exact rules are added by the user. Do not infer silently.

## Data Collection Advice

For next-week result:

Use X-ray-only downstream task:

```text
Normal vs Abnormal
```

If doing disease-specific, minimum meaningful target:

```text
100 positive + 100 negative
```

Better:

```text
150 positive + 150 negative
```

But current best first result is X-ray Normal vs Abnormal because X-ray has enough samples.

## Important Research Interpretation

Use this claim:

```text
Pilot X-ray normal/abnormal classification using Vietnamese hospital DICOM data,
doctor-conclusion-derived weak labels, and UniMiSSPlus transfer learning.
```

Avoid claiming:

```text
Clinical validation
Final diagnosis model
Disease-specific diagnosis across CT and X-ray
```

## Next Steps

1. Copy actual `LABELS.xlsx` into the project or update `LABELS.lnk`.
2. Run:

```powershell
python dicom_labeler.py extract LABELS.lnk DATA --output labels_raw_from_excel.csv
python dicom_labeler.py classify labels_raw_from_excel.csv --output labels_classified_from_excel.csv
```

3. Export X-ray images using:

```powershell
python dicom_analyzer.py export-unimissplus DATA <UniMiSSPlus_data_dir>
```

4. Build X-ray downstream lists:

```powershell
python dicom_labeler.py build-lists labels_classified_from_excel.csv DATA <UniMiSSPlus_data_dir> --output-dir labels
```

5. Train downstream X-ray normal/abnormal classifier.

6. Keep uncertain rows excluded unless manually reviewed.

