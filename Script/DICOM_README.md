# DICOM Patient Information Analyzer and Anonymizer

Analyzes and anonymizes DICOM files. Supports directories and ZIP archives.

## Installation

```bash
pip install pydicom
```

## Usage

### Analyze

```bash
python dicom_analyzer.py analyze <directory_or_zip> [--output report.json] [--no-recursive]
```

Reads ZIP files in-memory without extraction. Tracks unique patients, studies, modalities, body parts, demographics, and missing tags.

### Anonymize

```bash
python dicom_analyzer.py anonymize <input> <output>
```

**Default behavior:**
- Only keeps **CHEST** or **THORAX** scans (filters by `BodyPartExamined`)
- Removes direct names, free-text patient comments, physician/operator fields, dates/times, device/site fields
- Hashes linkable ID fields so internal lookback/mismatch checks are still possible
- Keeps `PatientSex` and `PatientAge` for post-analysis
- Uses parallel workers during anonymization for folders with multiple files/ZIPs
- **Folder input** -> folder output with full structure preserved
- **ZIP input** -> auto ZIP output with same filename
- **ZIP files inside folders** -> preserved as ZIP files, contents anonymized

### Clean Invalid ZIP Files

```bash
python dicom_analyzer.py clean-invalid-zips <directory_or_zip>
```

Deletes files ending in `.zip` that cannot actually be opened as ZIP archives. This is useful for files that produce errors like:

```text
Error reading ZIP file D:\DATA\CT\<id>.zip: File is not a zip file
```

Preview first without deleting:

```bash
python dicom_analyzer.py clean-invalid-zips <directory_or_zip> --dry-run
```

Save a cleanup report:

```bash
python dicom_analyzer.py clean-invalid-zips <directory_or_zip> --output cleanup_report.json
```

### Export UniMiSSPlus Format

```bash
python dicom_analyzer.py export-unimissplus <input> <output_data_dir>
```

Creates an upstream UniMiSSPlus-style data directory:

```text
<output_data_dir>/
  2D_images/*.png
  3D_images/*.nii.gz
  3D_subvolumes/*.nii.gz
  2D_images.txt
  3D_images.txt
  DRR_REQUIRED.txt
```

The exported files match the upstream loaders in `UniMiSSPlus/data_loader2D.py` and `UniMiSSPlus/data_loader3D2D.py`:

- X-ray DICOM (`DX`, `CR`, `XR`, `RG`) -> resized 512x512 PNG in `2D_images/`
- CT DICOM series -> NIfTI volume in `3D_images/`
- CT volume -> 24-slice subvolumes in `3D_subvolumes/` by default
- `2D_images.txt` and `3D_images.txt` are generated automatically
- `DRR_REQUIRED.txt` is written as a reminder that official DRR PNG generation is mandatory

UniMiSSPlus file roles:

| Path | Created by | Used for |
|---|---|---|
| `2D_images/*.png` | X-ray DICOM export | Real 2D X-ray SSL stream loaded by `Dataset2D` |
| `2D_images.txt` | `export-unimissplus` | List of real X-ray PNGs |
| `3D_images/*.nii.gz` | CT DICOM series export | Full CT volumes; source for subvolume extraction |
| `3D_subvolumes/*.nii.gz` | `export-unimissplus` | 24-slice CT chunks loaded by `Dataset3D2D` |
| `3D_subvolumes/*.png` | Official `pycuda_drr/rendering_DL.py` | DRR image paired with each CT subvolume |
| `3D_images.txt` | `export-unimissplus` | List of `3D_subvolumes/*.nii.gz`, not full `3D_images/*.nii.gz` |
| `DRR_REQUIRED.txt` | `export-unimissplus` | Reminder/instructions for required DRR generation |

CT series filtering:

- Scout/localizer/topogram series are skipped. This includes series whose metadata contains `LOCALIZER`, `SCOUT`, `TOPOGRAM`, `SURVIEW`, `HINH DINH VI`, or `DINH VI`.
- Screen-save/screenshot derived series are skipped.
- CT series with fewer than 24 slices are skipped by default, because UniMiSSPlus subvolumes are 24 slices deep.
- Different CT reconstructions, such as 1.25 mm and 5 mm axial series, are kept as separate NIfTI volumes because they have different `SeriesInstanceUID` values. They are not merged.
- Use `--min-ct-slices <N>` to change the minimum slice count.

How CT studies are interpreted:

- A CT study often starts with a scout/localizer series, shown in viewers as `Scout`, `HINH DINH VI`, or similar. These are positioning/topogram images, not volumetric CT stacks, so they are excluded from UniMiSSPlus export.
- The actual CT data usually appears as one or more axial volume series after the scout, for example `TRUOC TIEM`, `5mm TRUNG THAT`, `1.25mm NHU MO PHOI`, or similar.
- Each axial reconstruction is handled independently. A 1.25 mm lung reconstruction and a 5 mm mediastinum reconstruction are exported as separate CT volumes and separate sets of subvolumes.
- The exporter groups CT slices by `SeriesInstanceUID`. It does not merge different thicknesses, kernels, phases, or reconstructions into one volume.
- The number of slices can differ between series. This is expected. A thick 5 mm series may produce fewer 24-slice subvolumes; a thin 1.25 mm series may produce many more.
- UniMiSSPlus upstream training does not consume the full CT scan directly. It consumes the generated 24-slice subvolumes listed in `3D_images.txt`.

Optional export dependencies are required only for this command:

```bash
pip install numpy Pillow nibabel
```

Important: official DRR generation is **required** before UniMiSSPlus upstream pretraining. `export-unimissplus` intentionally does **not** create fake or mean-projection PNGs for CT subvolumes. After export, run the official DRR pipeline in a Linux/CUDA environment:

```bash
cd UniMiSSPlus/pycuda_drr
python setup.py install
python rendering_DL.py
```

Do not start upstream pretraining until every `3D_subvolumes/*.nii.gz` has a matching `3D_subvolumes/*.png` with the same stem. `UniMiSSPlus/data_loader3D2D.py` expects that paired PNG.

Verify after DRR:

```bash
python dicom_analyzer.py verify-unimissplus <output_data_dir>
```

This command fails if any list entry is missing or any CT subvolume lacks its required DRR PNG pair.

This export does not create downstream classification/segmentation labels. Fine-tuning still needs task labels or masks in the format required by each downstream script.

## Label Doctor Conclusions for Downstream Fine-Tuning

Use `dicom_labeler.py` after doctor conclusions have been collected in `LABELS.xlsx` or the shortcut `LABELS.lnk`.

The labeler does not diagnose images. It converts existing doctor report text into weak ML labels and keeps uncertain cases out of training by default.

### 1. Match Labels to DICOM ZIPs

```bash
python dicom_labeler.py extract LABELS.lnk DATA --output labels_raw.csv
```

Current project behavior:

- `LABELS.xlsx` column 1 is the external study/archive id.
- `LABELS.xlsx` column 2 is the doctor report/conclusion text.
- `LABELS.xlsx` column 3 is optional notes.
- Rows with an empty doctor report/conclusion are treated as default Normal samples.
- Label rows whose detail text contains `CT` are treated as CT labels; all other nonempty label rows are treated as X-ray labels.
- The current label ids match ZIP filename stems, so `dicom_labeler.py` falls back from `StudyInstanceUID` matching to ZIP-stem matching.

### 2. Check Label/Data Coverage

```bash
python dicom_labeler.py check-label-data LABELS.lnk DATA
```

Checks whether IDs mentioned in `LABELS.xlsx` exist as `<id>.zip` files in the DATA folder. This command focuses on the important direction for this project:

```text
LABELS.xlsx row -> DATA/<modality>/<id>.zip should exist
```

DATA ZIP files whose filename IDs do not appear in any `LABELS.xlsx` row are also reported by default as DATA files missing a label row. This is different from label rows with an empty detail column; empty detail rows are still valid default-normal samples.

Check only one modality:

```bash
python dicom_labeler.py check-label-data LABELS.lnk DATA --modality x-ray
python dicom_labeler.py check-label-data LABELS.lnk DATA --modality ct
```

Save CSV reports:

```bash
python dicom_labeler.py check-label-data LABELS.lnk DATA --modality x-ray --output-dir label_data_check_xray
python dicom_labeler.py check-label-data LABELS.lnk DATA --modality ct --output-dir label_data_check_ct
```

The report includes:

- label rows missing a matching DATA ZIP
- missing label rows categorized as CT or X-ray
- DATA ZIP files whose IDs do not appear in any label-file row
- duplicate label IDs with Excel row numbers and text previews
- duplicate DATA ZIP stems

### 3. Convert Conclusions to Weak Labels

```bash
python dicom_labeler.py classify labels_raw.csv --output labels_classified.csv
```

To classify every row in `LABELS.xlsx` without requiring a matching DICOM ZIP:

```bash
python dicom_labeler.py classify-xlsx LABELS.xlsx --output labels_all_classified.csv
```

To summarize common report phrases and export uncertain rows for manual review:

```bash
python dicom_labeler.py phrase-report labels_all_classified.csv --output-dir phrase_report
```

Default classification uses conservative Vietnamese report rules and writes:

- `coarse_label`: `NORMAL`, `ABNORMAL`, or `UNCERTAIN`
- `disease_label`: broad chest finding category
- `multi_labels`: all broad findings found by the rules
- `normal_abnormal_label`: `Abnormal`, `Normal`, or `EXCLUDE`
- `normal_abnormal_class`: `0` for abnormal, `1` for normal
- `confidence`, `evidence`, `needs_review`, and `label_reason`

Rows marked `UNCERTAIN` or `needs_review=yes` should be reviewed manually before being used for training. They are excluded from generated training lists by default.

Optional LLM-assisted labeling:

```bash
python dicom_labeler.py classify labels_raw.csv --method llm --output labels_classified.csv
```

Set `OPENAI_API_KEY` or pass `--api-key`. The LLM output is still weak labeling and should be reviewed, especially for low-confidence or uncertain rows.

### 4. Build Normal/Abnormal 2D Downstream Lists

After exporting images with `dicom_analyzer.py export-unimissplus`, build fixed split files for a first-stage normal/abnormal 2D classification task:

```bash
python dicom_labeler.py build-lists labels_classified.csv DATA <UniMiSSPlus_data_dir> --output-dir labels
```

This creates:

```text
labels/
  normal_abnormal_train.txt
  normal_abnormal_test.txt
  normal_abnormal_manifest.csv
```

List format:

```text
2D_images/<image>.png 0   # Abnormal
2D_images/<image>.png 1   # Normal
```

The split is study-level, not image-level, to reduce data leakage.

### 5. Build Stratified Grouped 5-Fold CV Lists

For the main Vietnamese X-ray study, use cross-validation instead of one fixed split:

```bash
python dicom_labeler.py build-cv-lists labels_all_classified.csv DATA <UniMiSSPlus_data_dir> --output-dir labels/vietnam_xray_cv
```

Default behavior:

- Builds `fold_0` through `fold_4`.
- Groups by `label_match_id`, so one study/archive ID does not appear in train and test in the same fold.
- Stratifies by `Normal` / `Abnormal` as much as possible while preserving groups.
- Uses only X-ray rows for this downstream task; CT rows are excluded.
- Excludes `UNCERTAIN`, `EXCLUDE`, and `needs_review=yes` rows unless `--include-review` is explicitly used.
- Repeats Abnormal rows 3x in `train_oversampled.txt` only.
- Keeps `val.txt` and `test.txt` natural, without oversampling.

Output:

```text
labels/vietnam_xray_cv/
  cv_manifest.csv
  cv_split_summary.csv
  fold_0/
    train.txt
    train_oversampled.txt
    val.txt
    test.txt
  ...
  fold_4/
    train.txt
    train_oversampled.txt
    val.txt
    test.txt
```

List label mapping remains:

```text
2D_images/<image>.png 0   # Abnormal, positive class for evaluation
2D_images/<image>.png 1   # Normal
```

Optional controls:

```bash
python dicom_labeler.py build-cv-lists labels_all_classified.csv DATA <UniMiSSPlus_data_dir> \
  --folds 5 \
  --val-fraction 0.15 \
  --abnormal-repeat 3 \
  --seed 2026 \
  --output-dir labels/vietnam_xray_cv
```

### 6. Evaluate Pretrained Frozen Features

Use this path for a fast comparison study without random CNN initialization. The models are public pretrained encoders; only a logistic-regression classifier is fitted inside each fold.

Recommended Linux GPU environment for RTX 5060 Ti 16GB:

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install scikit-learn matplotlib pillow numpy torchxrayvision transformers
```

Optional MedSigLIP baseline:

```bash
pip install transformers
```

Run the default pretrained-feature comparison:

```bash
python eval_pretrained_features.py \
  --data-root <UniMiSSPlus_data_dir> \
  --cv-dir labels/vietnam_xray_cv \
  --output-dir results/pretrained_feature_eval \
  --device auto \
  --batch-size 32
```

Default methods:

- `rad_dino`: recent CXR/biomedical self-supervised vision encoder from Microsoft.
- `torchxrayvision_densenet121`: CXR-specific DenseNet121 pretrained through TorchXRayVision.
- `imagenet_densenet121`: ImageNet DenseNet121 generic baseline.
- `imagenet_resnet50`: ImageNet ResNet50 generic baseline.

The ImageNet and TorchXRayVision methods are kept as comparison baselines, not as claims of current SOTA. The modern frozen-feature comparison should include `rad_dino`; if setup time allows, also consider CXR Foundation / MedImageInsight outside this script.

UniMiSSPlus fine-tuning is still the main project method, but use the same fold structure carefully: `train.txt` or `train_oversampled.txt` for optimization, `val.txt` for epoch/model selection, and `test.txt` only once for final fold reporting. Do not select the best epoch using the test fold.

Optional method list:

```bash
python eval_pretrained_features.py \
  --data-root <UniMiSSPlus_data_dir> \
  --cv-dir labels/vietnam_xray_cv \
  --methods rad_dino,torchxrayvision_densenet121,imagenet_densenet121,imagenet_resnet50,medsiglip \
  --output-dir results/pretrained_feature_eval
```

Output:

```text
results/pretrained_feature_eval/
  feature_eval_metrics.csv
  feature_eval_predictions.csv
  feature_eval_run.json
  feature_cache/*.npz
```

Important evaluation details:

- Abnormal is treated as the positive class.
- `train_oversampled.txt` is used by default for classifier fitting.
- Validation selects the logistic-regression `C` value.
- Test folds are never oversampled.
- Primary metrics are AUC, average precision, balanced accuracy, Abnormal recall, and Normal specificity.
- Do not report accuracy alone because the Normal/Abnormal dataset is imbalanced.

Summarize results:

```bash
python summarize_cv_results.py \
  --metrics results/pretrained_feature_eval/feature_eval_metrics.csv \
  --predictions results/pretrained_feature_eval/feature_eval_predictions.csv \
  --output-dir results/cv_summary
```

Summary output:

```text
results/cv_summary/
  cv_summary.csv
  cv_summary.md
  pooled_predictions.csv
  pooled_confusion_matrices.csv
  pooled_roc.png
  pooled_pr.png
```

Every method in the final table must have a citation entry in `study_references.md`.

### 7. CT / DRR Extension

CT is not used for the first downstream X-ray-only Normal/Abnormal claim. Keep CT as an extension:

- Export CT into 24-slice subvolumes with `export-unimissplus`.
- Generate official DRR PNGs with `UniMiSSPlus/pycuda_drr/rendering_DL.py`.
- Verify every CT subvolume has a paired DRR PNG with `verify-unimissplus`.
- Later compare X-ray-only training against X-ray plus CT-derived DRR augmentation.
- Do not claim CT improves performance unless patient/study leakage is controlled across all real X-ray and CT-derived DRR data.

## What is Removed / Replaced

| Tag | Action |
|---|---|
| **PatientName** | **Removed** |
| **PatientID** | **Replaced** with SHA-256 hash (16 chars) for lookback |
| **PatientBirthDate** | **Removed** |
| **PatientSex** | Kept |
| **PatientAge** | Kept |
| **PatientWeight** | **Removed** |
| **PatientAddress** | **Removed** |
| **OtherPatientIDs** | **Removed** |
| **OtherPatientNames** | **Removed** |
| **PatientComments** | **Removed** |
| **StudyDate** | **Removed** |
| **StudyTime** | **Removed** |
| **SeriesDate** | **Removed** |
| **SeriesTime** | **Removed** |
| **AcquisitionDate** | **Removed** |
| **AcquisitionTime** | **Removed** |
| **ContentDate** | **Removed** |
| **ContentTime** | **Removed** |
| **StudyID** | **Replaced** with SHA-256 hash (16 chars) for lookback |
| **AccessionNumber** | **Replaced** with SHA-256 hash (16 chars) for lookback |
| **StudyInstanceUID** | **Replaced** with deterministic DICOM-safe hashed UID |
| **SeriesInstanceUID** | **Replaced** with deterministic DICOM-safe hashed UID |
| **SOPInstanceUID** | **Replaced** with deterministic DICOM-safe hashed UID |
| **FrameOfReferenceUID** | **Replaced** with deterministic DICOM-safe hashed UID |
| **MediaStorageSOPInstanceUID** | **Replaced** with deterministic DICOM-safe hashed UID |
| **Modality** | Kept |
| **StudyDescription** | Kept |
| **SeriesDescription** | Kept |
| **InstitutionName** | Replaced with `ANONYMOUS` |
| **InstitutionAddress** | **Removed** |
| **InstitutionalDepartmentName** | **Removed** |
| **DeviceSerialNumber** | **Removed** |
| **StationName** | **Removed** |
| **ProtocolName** | **Removed** |
| Referring/performing/requesting physician and operator name tags | **Removed** |

## Notes

- Original files are **NEVER modified** - all output goes to the specified output directory/ZIP
- Anonymization uses up to `min(32, os.cpu_count())` worker threads for top-level folder work
- Hashed IDs are pseudonyms for internal QA/lookback; this is safer than raw IDs, but still not the same as a formal legal de-identification review
- Pixel data with burned-in patient info is **NOT** modified
- To customize behavior, edit `_anonymize_file()` in `dicom_analyzer.py`
