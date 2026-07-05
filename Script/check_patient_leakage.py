"""Check whether the same (hashed) PatientID appears across multiple study ZIPs.

Reads one DICOM file per ZIP (in-memory, no extraction) and records the
PatientID tag, which the existing anonymizer already replaces with a
deterministic SHA-256-derived hash -- same original patient => same hash.
"""
import sys
import zipfile
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import pydicom


def first_patient_id(zip_path: Path):
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith('/') or name.startswith('__MACOSX'):
                    continue
                try:
                    data = zf.read(name)
                except Exception:
                    continue
                try:
                    ds = pydicom.dcmread(BytesIO(data), force=False, stop_before_pixels=True)
                except Exception:
                    try:
                        ds = pydicom.dcmread(BytesIO(data), force=True, stop_before_pixels=True)
                    except Exception:
                        continue
                pid = ds.get('PatientID', None)
                if pid:
                    return str(pid)
    except Exception as e:
        return f"__ERROR__:{e}"
    return None


def scan(folder: Path, modality: str, patient_map: dict):
    zips = sorted(folder.glob('*.zip'))
    for i, z in enumerate(zips):
        pid = first_patient_id(z)
        patient_map[z.stem] = (modality, pid)
        if (i + 1) % 100 == 0 or (i + 1) == len(zips):
            print(f"  [{i+1}/{len(zips)}] {modality} scanned", file=sys.stderr)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('Script/ANONYMIZE')
    patient_map = {}
    scan(root / 'CT', 'CT', patient_map)
    scan(root / 'X-ray', 'X-ray', patient_map)

    by_patient = defaultdict(list)
    missing = []
    errors = []
    for study_id, (modality, pid) in patient_map.items():
        if pid is None:
            missing.append((study_id, modality))
        elif isinstance(pid, str) and pid.startswith('__ERROR__'):
            errors.append((study_id, modality, pid))
        else:
            by_patient[pid].append((study_id, modality))

    multi = {pid: studies for pid, studies in by_patient.items() if len(studies) > 1}

    print(f"\nTotal studies scanned: {len(patient_map)}")
    print(f"Studies with unreadable/missing PatientID: {len(missing)}")
    print(f"Studies with read errors: {len(errors)}")
    print(f"Unique patients (by hashed PatientID): {len(by_patient)}")
    print(f"Patients with >1 study (train/test leakage risk if grouped by study only): {len(multi)}")

    total_studies_in_multi = sum(len(v) for v in multi.values())
    print(f"Studies involved in a multi-study patient: {total_studies_in_multi}")

    out_csv = Path('patient_study_map.csv')
    with out_csv.open('w', encoding='utf-8') as f:
        f.write("study_id,modality,patient_hash\n")
        for study_id, (modality, pid) in sorted(patient_map.items()):
            f.write(f"{study_id},{modality},{pid}\n")
    print(f"\nWrote per-study patient hash map to {out_csv}")

    if multi:
        multi_csv = Path('patient_multi_study.csv')
        with multi_csv.open('w', encoding='utf-8') as f:
            f.write("patient_hash,study_id,modality\n")
            for pid, studies in sorted(multi.items(), key=lambda kv: -len(kv[1])):
                for study_id, modality in studies:
                    f.write(f"{pid},{study_id},{modality}\n")
        print(f"Wrote multi-study patient list to {multi_csv}")
        print("\nExample multi-study patients:")
        for pid, studies in list(multi.items())[:10]:
            print(f"  {pid}: {studies}")

    if missing:
        print("\nExample studies with missing PatientID (first 10):")
        for study_id, modality in missing[:10]:
            print(f"  {modality}/{study_id}")


if __name__ == '__main__':
    main()
