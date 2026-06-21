# AGENTS.md — Script/

Standalone DICOM analyzer and anonymizer. No relation to `UniMiSS/` or `UniMiSSPlus/`.

## Running
```powershell
.\venv\Scripts\Activate.ps1
python dicom_analyzer.py analyze <path> [--output report.json] [--no-recursive]
python dicom_analyzer.py anonymize <input> <output>
```
Only dependency: `pydicom` (pre-installed in `venv/`).

## Hardcoded behaviors (no CLI flags)

- **CHEST/THORAX filter only** — `BodyPartExamined` must contain `CHEST` or `THORAX` (case-insensitive). Everything else is skipped.
- **Minimal anonymization** — only removes: PatientName → `ANONYMOUS`, InstitutionName → `ANONYMOUS`, InstitutionAddress, InstitutionalDepartmentName, physician tags. All other tags (PatientID, dates, demographics) preserved.
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
