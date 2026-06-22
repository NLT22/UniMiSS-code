# AGENTS.md — Script/

Active DICOM analyzer, anonymizer, UniMiSSPlus exporter, verifier, and label helper.

## Running
```powershell
.\venv\Scripts\Activate.ps1
python dicom_analyzer.py analyze <path> [--output report.json] [--no-recursive]
python dicom_analyzer.py anonymize <input> <output>
python dicom_analyzer.py export-unimissplus <input> <output_data_dir>
python dicom_analyzer.py verify-unimissplus <output_data_dir>
python dicom_labeler.py extract LABELS.lnk DATA --output labels_raw.csv
python dicom_labeler.py classify labels_raw.csv --output labels_classified.csv
python dicom_labeler.py build-lists labels_classified.csv DATA <UniMiSSPlus_data_dir> --output-dir labels
```
Core dependency: `pydicom` (pre-installed in `venv/`). Export also needs `numpy`, `Pillow`, and `nibabel`; labeling needs `openpyxl`.

## Hardcoded behaviors (no CLI flags)

- **CHEST/THORAX filter only** — `BodyPartExamined` must contain `CHEST` or `THORAX` (case-insensitive). Everything else is skipped.
- **Anonymization** — preserves `PatientAge` and `PatientSex`; hashes linkable IDs/UIDs; removes direct identifiers, dates/times, patient free-text, institution/site/device fields, and physician/operator tags.
- **ZIP input → ZIP output**, **folder input → folder output**.
- **ZIPs inside folders stay as ZIPs** — extracted internally, anonymized, re-zipped with original name.
- **All-filtered ZIPs are NOT created** — if no CHEST/THORAX files remain, the output ZIP is omitted entirely.
- **Original files are never modified** — all output to destination only.

## DICOM reading
- Always try `force=False` first, fallback to `force=True` on failure.
- Prefer `ds.get('TagName')` for safe tag access (not `getattr`, not `ds.TagName`).

## ZIP handling
- `analyze` mode: reads entries in-memory via `BytesIO` (no disk extraction).
- `anonymize` mode: extracts to `tempfile.TemporaryDirectory`, processes, optionally re-zips.

## Patient folder sanitization (ZIP entries)
Strips `-PatientName` suffix from **first** path component only:
`123123124-JohnDoe/file.dcm` → `123123124/file.dcm`. Subfolders inside are not touched.

## UniMiSSPlus export
- X-ray DICOM (`DX`, `CR`, `XR`, `RG`) -> 512x512 PNG in `2D_images/`.
- CT DICOM series -> NIfTI volume in `3D_images/`.
- CT volume -> 24-slice subvolumes in `3D_subvolumes/`.
- `2D_images.txt` lists real X-ray PNGs.
- `3D_images.txt` lists CT subvolume NIfTI files, not full CT volumes.
- Official DRR generation must later create a matching `3D_subvolumes/*.png` for every CT subvolume.
