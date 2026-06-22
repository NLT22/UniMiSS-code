# AGENTS.md — UniMiSS-code

## Repo structure

Two project areas remain in this repo:

| Directory | Project | Tech |
|-----------|---------|------|
| `Script/` | DICOM analyzer/anonymizer/exporter/labeler | Python 3.10, pydicom |
| `UniMiSSPlus/` | UniMiSS+ (TPAMI) frozen reference | PyTorch 1.8.1, CUDA 11.1, Python 3.7 |

No test framework, no CI, no linter config anywhere in the repo.

## Script/ — active DICOM tooling

The only actively maintained code. It prepares hospital DICOM chest CT/X-ray data for UniMiSSPlus-style upstream/domain-adaptation use and downstream X-ray-only classification.

Main files:
- `dicom_analyzer.py`: analyze, anonymize, export UniMiSSPlus format, verify UniMiSSPlus export.
- `dicom_labeler.py`: link `LABELS.xlsx` reports to DICOM studies, create weak labels, build downstream normal/abnormal lists.
- `DICOM_README.md`: current operating notes.
- `TRANSFER_SUMMARY.md`: current project handoff summary.

### Key behaviors
- Filters to **CHEST** or **THORAX** only (`BodyPartExamined` tag).
- CT scout/localizer/topogram and screenshot-derived series are skipped during export.
- CT series are grouped by `SeriesInstanceUID`; different slice thicknesses/reconstructions are kept separate.
- **ZIP input → ZIP output**, **folder input → folder output**.
- ZIP files inside folders stay as ZIPs (extracted internally, re-zipped).
- All-filtered ZIPs (no CHEST/THORAX) are **not** created.
- Original files are never modified.

### Anonymization
- Preserve: `PatientAge`, `PatientSex`.
- Hash linkable IDs/UIDs such as `PatientID`, `StudyID`, `AccessionNumber`, `StudyInstanceUID`, `SeriesInstanceUID`, and `SOPInstanceUID`.
- Remove or replace direct identifiers, dates/times, patient free-text, institution/site/device fields, and physician/operator tags.
- Pixel data with burned-in text is not modified.

### DICOM reading
- Always try `force=False` first, fallback to `force=True` on failure.
- Use `ds.get('TagName')` for safe tag access (not `getattr`).

### ZIP handling
- `analyze` mode: reads ZIP entries in-memory via `BytesIO` (no extraction).
- `anonymize` mode: extracts ZIPs to `tempfile.TemporaryDirectory`, processes, re-zips.

### Patient folder sanitization
Strips `-PatientName` suffix from **first** path component only:
`123123124-JohnDoe/file.dcm` → `123123124/file.dcm`. Subfolders are preserved.

### Running
```powershell
# Activate venv first
.\Script\venv\Scripts\Activate.ps1

python Script\dicom_analyzer.py analyze <path> [--output report.json]
python Script\dicom_analyzer.py anonymize <input> <output>
python Script\dicom_analyzer.py export-unimissplus <input> <output_data_dir>
python Script\dicom_analyzer.py verify-unimissplus <output_data_dir>
```

Core dependency: `pydicom` (pre-installed in `Script/venv/`). Export also needs `numpy`, `Pillow`, and `nibabel`; labeling needs `openpyxl`.

## UniMiSSPlus — frozen research code

Kept as a reference and target folder layout for exported data. Expect:
- Pinned old PyTorch version (1.8.1), Python 3.7, CUDA 11.1.
- No working conda envs on this machine.
- No dataset files present (all in `.gitignore`).
- No automated tests.
- Pretrained weights hosted on Google Drive (links in READMEs).

Official DRR generation is still required before upstream pretraining:
```bash
cd UniMiSSPlus/pycuda_drr
python setup.py install
python rendering_DL.py
```

Do not modify `UniMiSSPlus/` unless explicitly asked; it exists for reference and later Linux/CUDA work.
