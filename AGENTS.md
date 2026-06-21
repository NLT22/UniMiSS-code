# AGENTS.md — UniMiSS-code

## Repo structure

Three independent subprojects in one repo (no shared config):

| Directory | Project | Tech |
|-----------|---------|------|
| `Script/` | DICOM analyzer/anonymizer | Python 3.10, pydicom |
| `UniMiSS/` | UniMiSS (ECCV 2022) | PyTorch 1.7.1, CUDA 11.0, Python 3.7 |
| `UniMiSSPlus/` | UniMiSS+ (TPAMI) | PyTorch 1.8.1, CUDA 11.1, Python 3.7 |

No test framework, no CI, no linter config anywhere in the repo.

## Script/ — DICOM analyzer (`dicom_analyzer.py`)

The only actively maintained code. A standalone tool — no relation to the ML projects.

### Key behaviors (hardcoded, no CLI flags)
- Filters to **CHEST** or **THORAX** only (`BodyPartExamined` tag).
- Removes only: PatientName → `ANONYMOUS`, InstitutionName → `ANONYMOUS`, InstitutionAddress, InstitutionalDepartmentName, physician tags.
- Preserves all other tags (PatientID, dates, demographics).
- **ZIP input → ZIP output**, **folder input → folder output**.
- ZIP files inside folders stay as ZIPs (extracted internally, re-zipped).
- All-filtered ZIPs (no CHEST/THORAX) are **not** created.
- Original files are never modified.

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
```

Only dependency: `pydicom` (pre-installed in `Script/venv/`).

## UniMiSS / UniMiSSPlus — not maintained

Both are frozen research codebases. Expect:
- Pinned old PyTorch versions (1.7.1 / 1.8.1), Python 3.7, CUDA 11.x.
- No working conda envs on this machine.
- No dataset files present (all in `.gitignore`).
- No automated tests.
- Pretrained weights hosted on Google Drive (links in READMEs).

### Standard workflow (historical reference)
```bash
conda create -n unimiss python=3.7
conda activate unimiss
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
cd UniMiSS && sh run.sh
```

Do not modify these unless explicitly asked — they exist for reference only.
