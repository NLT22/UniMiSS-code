#!/usr/bin/env python3
"""
DICOM Patient Information Analyzer, Anonymizer, and UniMiSSPlus Exporter

Usage:
    python dicom_analyzer.py analyze <directory> [--output report.json] [--no-recursive]
    python dicom_analyzer.py anonymize <input_dir> <output_dir>
    python dicom_analyzer.py clean-invalid-zips <directory> [--dry-run]
    python dicom_analyzer.py export-unimissplus <input> <output_data_dir>
    python dicom_analyzer.py verify-unimissplus <output_data_dir>

The UniMiSSPlus export path converts CHEST/THORAX DICOM data into the
upstream SSL folder layout expected by UniMiSSPlus. CT scout/localizer
series are skipped, axial CT series are converted to NIfTI subvolumes,
and official DRR generation is required afterward to create paired PNGs.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict
import json
import zipfile
import tempfile
from io import BytesIO
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
import math

try:
    import pydicom
except ImportError:
    print("Error: pydicom is not installed. Install it with: pip install pydicom")
    sys.exit(1)


class DICOMAnalyzer:
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'valid_dicom': 0,
            'invalid_files': 0,
            'patients': defaultdict(int),
            'institutions': Counter(),
            'modalities': Counter(),
            'study_dates': Counter(),
            'body_parts': Counter(),
            'missing_tags': Counter(),
            'file_errors': []
        }
        self.unique_patients = set()
        self.unique_studies = set()
        self.patient_demographics = {}

    def analyze_directory(self, directory: str, recursive: bool = True) -> Dict:
        print(f"Analyzing DICOM files in: {directory}")

        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if path.is_file() and path.suffix.lower() == '.zip':
            self._analyze_zip_file(path)
        else:
            pattern = "**/*" if recursive else "*"
            files = list(path.glob(pattern))

            for file_path in files:
                if file_path.is_file():
                    if file_path.suffix.lower() == '.zip':
                        self._analyze_zip_file(file_path)
                    else:
                        self.stats['total_files'] += 1
                        self._analyze_file(file_path)

                    if self.stats['total_files'] % 100 == 0:
                        print(f"Processed {self.stats['total_files']} files...")

        return self._compile_report()

    def _analyze_zip_file(self, zip_path: Path):
        print(f"Reading ZIP file: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()

                for file_name in file_list:
                    if file_name.endswith('/'):
                        continue

                    self.stats['total_files'] += 1

                    try:
                        with zip_ref.open(file_name) as file_data:
                            file_bytes = BytesIO(file_data.read())
                            self._analyze_file_from_buffer(file_bytes, f"{zip_path.name}:{file_name}")
                    except Exception as e:
                        self.stats['invalid_files'] += 1
                        self.stats['file_errors'].append({
                            'file': f"{zip_path}:{file_name}",
                            'error': str(e)
                        })

                    if self.stats['total_files'] % 100 == 0:
                        print(f"Processed {self.stats['total_files']} files...")

        except Exception as e:
            print(f"Error reading ZIP file {zip_path}: {e}")

    def _analyze_file(self, file_path: Path):
        try:
            try:
                ds = pydicom.dcmread(str(file_path), force=False)
            except Exception:
                ds = pydicom.dcmread(str(file_path), force=True)

            self.stats['valid_dicom'] += 1
            self._extract_metadata(ds)

        except Exception as e:
            self.stats['invalid_files'] += 1
            self.stats['file_errors'].append({
                'file': str(file_path),
                'error': str(e)
            })

    def _analyze_file_from_buffer(self, file_buffer, file_name: str):
        try:
            try:
                ds = pydicom.dcmread(file_buffer, force=False)
            except Exception:
                file_buffer.seek(0)
                ds = pydicom.dcmread(file_buffer, force=True)

            self.stats['valid_dicom'] += 1
            self._extract_metadata(ds)

        except Exception as e:
            self.stats['invalid_files'] += 1
            self.stats['file_errors'].append({
                'file': file_name,
                'error': str(e)
            })

    def _extract_metadata(self, ds):
        patient_id = ds.get('PatientID', None)

        if patient_id:
            self.stats['patients'][patient_id] += 1
            self.unique_patients.add(patient_id)

        institution = ds.get('InstitutionName', None)
        if institution:
            self.stats['institutions'][institution] += 1
        else:
            self.stats['missing_tags']['InstitutionName'] += 1

        modality = ds.get('Modality', None)
        if modality:
            self.stats['modalities'][modality] += 1
        else:
            self.stats['modalities']['UNKNOWN'] += 1

        study_date = ds.get('StudyDate', None)
        if study_date:
            year = study_date[:4] if len(study_date) >= 4 else 'UNKNOWN'
            self.stats['study_dates'][year] += 1
        else:
            self.stats['missing_tags']['StudyDate'] += 1

        body_part = ds.get('BodyPartExamined', None)
        if body_part:
            self.stats['body_parts'][body_part] += 1
        else:
            self.stats['body_parts']['UNKNOWN'] += 1

        if patient_id and patient_id not in self.patient_demographics:
            sex = ds.get('PatientSex', None)
            age = ds.get('PatientAge', None)
            self.patient_demographics[patient_id] = {
                'sex': sex if sex else 'UNKNOWN',
                'age': age
            }

        study_uid = ds.get('StudyInstanceUID', None)
        if study_uid:
            self.unique_studies.add(study_uid)

    def _get_age_range(self, age_str: str) -> str:
        try:
            age_num = int(age_str.replace('Y', '').replace('M', '').replace('D', ''))
            if 'D' in age_str or 'M' in age_str:
                return '0-1'
            elif age_num < 10: return '0-10'
            elif age_num < 20: return '10-20'
            elif age_num < 30: return '20-30'
            elif age_num < 40: return '30-40'
            elif age_num < 50: return '40-50'
            elif age_num < 60: return '50-60'
            elif age_num < 70: return '60-70'
            else: return '70+'
        except:
            return 'UNKNOWN'

    def _compile_report(self) -> Dict:
        patient_sex_counter = Counter()
        patient_age_counter = Counter()

        for demographics in self.patient_demographics.values():
            patient_sex_counter[demographics['sex']] += 1
            if demographics['age']:
                patient_age_counter[self._get_age_range(demographics['age'])] += 1
            else:
                patient_age_counter['UNKNOWN'] += 1

        return {
            'summary': {
                'total_files_scanned': self.stats['total_files'],
                'valid_dicom_files': self.stats['valid_dicom'],
                'invalid_files': self.stats['invalid_files'],
                'unique_patients': len(self.unique_patients),
                'unique_studies': len(self.unique_studies),
            },
            'patient_distribution': dict(self.stats['patients']),
            'institutions': dict(self.stats['institutions']),
            'modalities': dict(self.stats['modalities']),
            'study_years': dict(self.stats['study_dates']),
            'body_parts': dict(self.stats['body_parts']),
            'patient_sex': dict(patient_sex_counter),
            'patient_age_ranges': dict(patient_age_counter),
            'missing_tags': dict(self.stats['missing_tags']),
            'errors': self.stats['file_errors'][:10]
        }

    def print_report(self, report: Dict):
        print("\n" + "="*80)
        print("DICOM DATASET ANALYSIS REPORT")
        print("="*80)

        print("\n--- SUMMARY ---")
        for key, value in report['summary'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\n--- INSTITUTIONS ---")
        for inst, count in sorted(report['institutions'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {inst}: {count} files")

        print("\n--- MODALITIES ---")
        for mod, count in sorted(report['modalities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {mod}: {count} files")

        print("\n--- STUDY YEARS ---")
        for year, count in sorted(report['study_years'].items()):
            print(f"  {year}: {count} files")

        print("\n--- BODY PARTS ---")
        for part, count in sorted(report['body_parts'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {part}: {count} files")

        print("\n--- PATIENT DEMOGRAPHICS ---")
        print("  Sex Distribution:")
        for sex, count in sorted(report['patient_sex'].items()):
            print(f"    {sex}: {count} patients")

        print("  Age Distribution:")
        for age_range, count in sorted(report['patient_age_ranges'].items()):
            print(f"    {age_range}: {count} patients")

        if report['missing_tags']:
            print("\n--- MISSING TAGS ---")
            for tag, count in sorted(report['missing_tags'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {tag}: {count} files")

        if report['errors']:
            print("\n--- ERRORS (First 10) ---")
            for error in report['errors']:
                print(f"  {error['file']}: {error['error']}")

        print("\n" + "="*80)


class DICOMAnonymizer:
    """Create anonymized CHEST/THORAX DICOM copies without touching originals."""

    def __init__(self):
        self.stats = {
            'processed': 0, 'anonymized': 0, 'skipped': 0, 'failed': 0, 'errors': []
        }
        self.max_workers = min(32, max(1, os.cpu_count() or 1))
        self._stats_lock = Lock()

    def _add_stat(self, key: str, amount: int = 1):
        with self._stats_lock:
            self.stats[key] += amount

    def _add_error(self, file_name: str, error: Exception):
        with self._stats_lock:
            self.stats['errors'].append({
                'file': file_name,
                'error': str(error)
            })

    def _hash_value(self, value, length: int = 16) -> str:
        return hashlib.sha256(str(value).encode()).hexdigest()[:length]

    def _hash_uid(self, value) -> str:
        # Keep UID values deterministic for lookback while staying valid DICOM UIDs.
        digest = hashlib.sha256(str(value).encode()).hexdigest()[:32]
        return f"2.25.{int(digest, 16)}"

    def _delete_tags(self, ds, tags):
        for tag in tags:
            if hasattr(ds, tag):
                delattr(ds, tag)

    def _hash_tags(self, ds, tags):
        for tag in tags:
            value = ds.get(tag, None)
            if value:
                setattr(ds, tag, self._hash_value(value))

    def _hash_uid_tags(self, ds, tags):
        for tag in tags:
            value = ds.get(tag, None)
            if value:
                setattr(ds, tag, self._hash_uid(value))

    def anonymize_directory(self, input_dir: str, output_dir: str):
        """Anonymize a folder or ZIP, preserving the input/output container shape."""
        print(f"Anonymizing DICOM files from {input_dir} to {output_dir}")
        print(f"NOTE: Original files will NOT be modified. Anonymized copies will be saved to output directory.\n")

        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if input_path.is_file() and input_path.suffix.lower() == '.zip':
            output_path = Path(output_dir)
            if output_path.suffix.lower() != '.zip':
                output_path = output_path / input_path.name
            print(f"Input is ZIP file - output will be saved as: {output_path}\n")

            with tempfile.TemporaryDirectory() as temp_output_dir:
                temp_output_path = Path(temp_output_dir)
                self._process_files(input_path, temp_output_path)

                files_in_zip = [f for f in temp_output_path.rglob('*') if f.is_file()]
                if not files_in_zip:
                    print("All files were filtered out (non-CHEST/THORAX) - no output ZIP created.\n")
                    self._print_summary()
                    return

                print(f"\nCreating output ZIP file: {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                    for file_path in files_in_zip:
                        arcname = file_path.relative_to(temp_output_path)
                        zip_out.write(file_path, arcname)
                print(f"Anonymized files saved to ZIP: {output_path}")
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self._process_files(input_path, output_path)

        self._print_summary()

    def _process_files(self, input_path: Path, output_path: Path, parallel: bool = True):
        """Process top-level files in parallel; process extracted ZIP contents sequentially."""
        if input_path.is_file() and input_path.suffix.lower() == '.zip':
            self._anonymize_zip_file(input_path, output_path)
        else:
            files = [file_path for file_path in input_path.glob("**/*") if file_path.is_file()]
            if len(files) <= 1 or not parallel:
                for file_path in files:
                    self._process_file_task(file_path, input_path, output_path)
                return

            print(f"Using {self.max_workers} parallel workers for anonymization.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_file_task, file_path, input_path, output_path)
                    for file_path in files
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self._add_stat('failed')
                        self._add_error(str(input_path), e)

    def _process_file_task(self, file_path: Path, input_path: Path, output_path: Path):
        relative_path = file_path.relative_to(input_path)
        if file_path.suffix.lower() == '.zip':
            out_zip = output_path / relative_path
            out_zip.parent.mkdir(parents=True, exist_ok=True)
            self._anonymize_zip_to_zip(file_path, out_zip)
            return

        self._add_stat('processed')
        out_file = output_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._anonymize_file(file_path, out_file):
            self._add_stat('processed', -1)

    def _anonymize_zip_to_zip(self, input_zip: Path, output_zip: Path):
        print(f"Processing ZIP file: {input_zip}")
        try:
            with tempfile.TemporaryDirectory() as temp_extract:
                with zipfile.ZipFile(input_zip, 'r') as zf:
                    zf.extractall(temp_extract)

                with tempfile.TemporaryDirectory() as temp_output:
                    self._process_files(Path(temp_extract), Path(temp_output), parallel=False)

                    files_in_zip = [f for f in Path(temp_output).rglob('*') if f.is_file()]
                    if not files_in_zip:
                        print(f"  All files filtered out (non-CHEST/THORAX) - skipping ZIP.\n")
                        return

                    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf_out:
                        for f in files_in_zip:
                            arcname = f.relative_to(Path(temp_output))
                            sanitized = self._sanitize_zip_path(str(arcname))
                            zf_out.write(f, sanitized)
        except Exception as e:
            print(f"Error processing ZIP file {input_zip}: {e}")

    def _anonymize_zip_file(self, zip_path: Path, output_dir: Path):
        print(f"Processing ZIP file: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()

                with tempfile.TemporaryDirectory() as temp_dir:
                    for file_name in file_list:
                        if file_name.endswith('/'):
                            continue

                        self._add_stat('processed')

                        try:
                            zip_ref.extract(file_name, temp_dir)
                            temp_file_path = Path(temp_dir) / file_name

                            sanitized = self._sanitize_zip_path(file_name)
                            out_file = output_dir / sanitized
                            out_file.parent.mkdir(parents=True, exist_ok=True)

                            if not self._anonymize_file(temp_file_path, out_file):
                                self._add_stat('processed', -1)

                        except Exception as e:
                            self._add_stat('failed')
                            self._add_error(f"{zip_path}:{file_name}", e)

        except Exception as e:
            print(f"Error reading ZIP file {zip_path}: {e}")

    def _sanitize_zip_path(self, file_name: str) -> str:
        """Strip only the patient-name suffix from the first ZIP path component."""
        parts = Path(file_name).parts
        if len(parts) <= 1:
            return file_name
        return str(Path(parts[0].split('-')[0], *list(parts[1:])))

    def _anonymize_file(self, input_file: Path, output_file: Path):
        """Anonymize one DICOM file and return False when it is filtered out."""
        try:
            try:
                ds = pydicom.dcmread(str(input_file), force=False)
            except Exception:
                ds = pydicom.dcmread(str(input_file), force=True)

            body_part = ds.get('BodyPartExamined', '')
            if 'CHEST' not in body_part.upper() and 'THORAX' not in body_part.upper():
                self._add_stat('skipped')
                return False

            # Patient identity: remove names/free text, hash lookback IDs, keep age/sex.
            if hasattr(ds, 'PatientName'):
                delattr(ds, 'PatientName')

            patient_id = ds.get('PatientID', None)
            if patient_id:
                ds.PatientID = self._hash_value(patient_id)

            birth_date = ds.get('PatientBirthDate', None)
            if birth_date:
                try:
                    age = ds.get('PatientAge', None)
                    if not age:
                        ref_date = ds.get('StudyDate', None)
                        if ref_date and len(ref_date) >= 8:
                            ref = datetime(int(ref_date[:4]), int(ref_date[4:6]), int(ref_date[6:8]))
                        else:
                            ref = datetime.now()
                        birth = datetime(int(birth_date[:4]), int(birth_date[4:6]), int(birth_date[6:8]))
                        years = ref.year - birth.year
                        if (ref.month, ref.day) < (birth.month, birth.day):
                            years -= 1
                        ds.PatientAge = f"{years:03d}Y"
                except Exception:
                    pass
                delattr(ds, 'PatientBirthDate')
            # PatientSex and PatientAge are intentionally kept for post-analysis.
            if hasattr(ds, 'PatientWeight'):
                delattr(ds, 'PatientWeight')
            if hasattr(ds, 'PatientAddress'):
                delattr(ds, 'PatientAddress')
            self._delete_tags(ds, [
                'OtherPatientIDs',
                'OtherPatientNames',
                'PatientComments',
            ])

            # Study/date identity: delete exact dates/times, hash linkable IDs.
            if hasattr(ds, 'StudyDate'):
                delattr(ds, 'StudyDate')
            if hasattr(ds, 'StudyTime'):
                delattr(ds, 'StudyTime')
            study_id = ds.get('StudyID', None)
            if study_id:
                ds.StudyID = self._hash_value(study_id)
            self._hash_tags(ds, ['AccessionNumber'])
            self._hash_uid_tags(ds, [
                'StudyInstanceUID',
                'SeriesInstanceUID',
                'SOPInstanceUID',
                'FrameOfReferenceUID',
            ])
            if hasattr(ds, 'file_meta') and ds.file_meta:
                media_storage_uid = ds.file_meta.get('MediaStorageSOPInstanceUID', None)
                if media_storage_uid:
                    ds.file_meta.MediaStorageSOPInstanceUID = self._hash_uid(media_storage_uid)
            self._delete_tags(ds, [
                'AcquisitionDate',
                'AcquisitionTime',
                'ContentDate',
                'ContentTime',
            ])
            # Modality, StudyDescription, and SeriesDescription are kept for analysis.

            # Series/device fields: delete date/time and scanner identifiers.
            if hasattr(ds, 'SeriesDate'):
                delattr(ds, 'SeriesDate')
            if hasattr(ds, 'SeriesTime'):
                delattr(ds, 'SeriesTime')
            self._delete_tags(ds, [
                'DeviceSerialNumber',
                'StationName',
                'ProtocolName',
            ])
            # Institution / physician fields: anonymize site and remove people names.
            if hasattr(ds, 'InstitutionName'):
                ds.InstitutionName = 'ANONYMOUS'
            if hasattr(ds, 'InstitutionAddress'):
                delattr(ds, 'InstitutionAddress')
            if hasattr(ds, 'InstitutionalDepartmentName'):
                delattr(ds, 'InstitutionalDepartmentName')
            for tag in ['PhysicianOfRecord', 'PerformingPhysicianName',
                        'NameOfPhysicianReadingStudy', 'OperatorName']:
                if hasattr(ds, tag):
                    delattr(ds, tag)
            self._delete_tags(ds, [
                'ReferringPhysicianName',
                'PhysiciansOfRecord',
                'NameOfPhysiciansReadingStudy',
                'OperatorsName',
                'RequestingPhysician',
            ])

            ds.save_as(str(output_file))
            self._add_stat('anonymized')
            return True

        except Exception as e:
            self._add_stat('failed')
            self._add_error(str(input_file), e)
            return False

    def _print_summary(self):
        print("\n" + "="*80)
        print("ANONYMIZATION SUMMARY")
        print("="*80)
        print(f"  Total files processed: {self.stats['processed']}")
        print(f"  Successfully anonymized: {self.stats['anonymized']}")
        if self.stats['skipped']:
            print(f"  Skipped (non-CHEST/THORAX): {self.stats['skipped']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"\n  NOTE: Original files were NOT modified.")
        print(f"  All anonymized files were saved to the output directory only.")
        print("="*80)


class InvalidZipCleaner:
    """Delete .zip files that cannot be opened as ZIP archives."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = Counter()
        self.deleted = []
        self.invalid = []
        self.valid = []
        self.errors = []

    def clean(self, input_path: str):
        root = Path(input_path)
        if not root.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        zip_files = self._collect_zip_files(root)
        print(f"Scanning {len(zip_files)} ZIP file(s) under: {root}")
        if self.dry_run:
            print("Dry run enabled: invalid ZIP files will be reported but not deleted.")

        for zip_path in zip_files:
            self._check_and_delete(zip_path)

        self._print_summary()
        return {
            "scanned": self.stats["scanned"],
            "valid": self.stats["valid"],
            "invalid": self.stats["invalid"],
            "deleted": self.stats["deleted"],
            "delete_failed": self.stats["delete_failed"],
            "dry_run": self.dry_run,
            "invalid_files": self.invalid,
            "deleted_files": self.deleted,
            "errors": self.errors,
        }

    def _collect_zip_files(self, root: Path):
        if root.is_file():
            return [root] if root.suffix.lower() == ".zip" else []
        return sorted(path for path in root.rglob("*.zip") if path.is_file())

    def _check_and_delete(self, zip_path: Path):
        self.stats["scanned"] += 1
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.namelist()
            self.stats["valid"] += 1
            self.valid.append(str(zip_path))
            return
        except zipfile.BadZipFile as e:
            reason = str(e) or "File is not a zip file"
        except zipfile.LargeZipFile as e:
            reason = str(e)
        except OSError as e:
            reason = str(e)

        self.stats["invalid"] += 1
        self.invalid.append(str(zip_path))
        print(f"Invalid ZIP: {zip_path} ({reason})")

        if self.dry_run:
            return

        try:
            zip_path.unlink()
            self.stats["deleted"] += 1
            self.deleted.append(str(zip_path))
            print(f"Deleted: {zip_path}")
        except Exception as e:
            self.stats["delete_failed"] += 1
            self.errors.append({"file": str(zip_path), "error": str(e)})
            print(f"Error deleting {zip_path}: {e}")

    def _print_summary(self):
        print("\n" + "="*80)
        print("INVALID ZIP CLEANUP SUMMARY")
        print("="*80)
        print(f"  Scanned ZIP files: {self.stats['scanned']}")
        print(f"  Valid ZIP files: {self.stats['valid']}")
        print(f"  Invalid ZIP files: {self.stats['invalid']}")
        if self.dry_run:
            print("  Deleted files: 0 (dry run)")
        else:
            print(f"  Deleted files: {self.stats['deleted']}")
        print(f"  Delete failures: {self.stats['delete_failed']}")
        if self.invalid:
            print("\n--- INVALID ZIP FILES (First 20) ---")
            for path in self.invalid[:20]:
                print(f"  {path}")
        if self.errors:
            print("\n--- DELETE ERRORS ---")
            for error in self.errors[:20]:
                print(f"  {error['file']}: {error['error']}")
        print("="*80)


class UniMiSSPlusExporter:
    """Export DICOM ZIP/folder data into the UniMiSSPlus upstream SSL layout.

    This command deliberately does not create synthetic CT projection PNGs.
    Official pycuda_drr generation must run after export so each CT subvolume
    has the physics-based DRR PNG expected by UniMiSSPlus/data_loader3D2D.py.
    """

    def __init__(self, subvolume_depth: int = 24, subvolume_stride: int = 12, min_ct_slices: int = 24):
        self.subvolume_depth = subvolume_depth
        self.subvolume_stride = subvolume_stride
        self.min_ct_slices = min_ct_slices
        self.stats = Counter()
        self.errors = []
        self.np = None
        self.Image = None
        self.nib = None

    def _load_deps(self):
        """Load pixel-export dependencies only when export-unimissplus is used."""
        missing = []
        try:
            import numpy as np
            self.np = np
        except ImportError:
            missing.append("numpy")
        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            missing.append("Pillow")
        try:
            import nibabel as nib
            self.nib = nib
        except ImportError:
            missing.append("nibabel")

        if missing:
            raise RuntimeError(
                "export-unimissplus needs optional pixel export dependencies: "
                + ", ".join(missing)
                + ". Install them in Script/venv with: pip install numpy Pillow nibabel"
            )

    def export(self, input_path: str, output_data_dir: str):
        """Create 2D_images, 3D_images, 3D_subvolumes, and SSL list files."""
        self._load_deps()

        source = Path(input_path)
        output = Path(output_data_dir)
        if not source.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        images2d_dir = output / "2D_images"
        images3d_dir = output / "3D_images"
        subvolumes_dir = output / "3D_subvolumes"
        images2d_dir.mkdir(parents=True, exist_ok=True)
        images3d_dir.mkdir(parents=True, exist_ok=True)
        subvolumes_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting UniMiSSPlus data from {source} to {output}")
        records = self._collect_records(source)
        ct_series = defaultdict(list)
        list_2d = []

        for record in records:
            ds = record["ds"]
            modality = str(ds.get("Modality", "")).upper()
            if modality in ("DX", "CR", "XR", "RG"):
                # Real X-ray stream for Dataset2D.
                rel_path = self._export_xray_png(record, images2d_dir)
                if rel_path:
                    list_2d.append(rel_path)
            elif modality == "CT":
                # Keep each CT reconstruction separate; do not merge 1.25 mm and 5 mm series.
                series_uid = str(ds.get("SeriesInstanceUID", ds.get("StudyInstanceUID", record["name"])))
                ct_series[series_uid].append(record)
            else:
                self.stats["skipped_unknown_modality"] += 1

        list_3d = []
        for records_in_series in ct_series.values():
            # Skip scouts/localizers/screen-saves before decoding all pixel data.
            skip_reason = self._ct_series_skip_reason(records_in_series)
            if skip_reason:
                self.stats[f"ct_series_skipped_{skip_reason}"] += 1
                continue

            nii_path = self._export_ct_series(records_in_series, images3d_dir)
            if nii_path:
                self.stats["ct_series_exported"] += 1
                list_3d.extend(self._export_subvolumes(nii_path, subvolumes_dir))

        self._write_list(output / "2D_images.txt", list_2d)
        self._write_list(output / "3D_images.txt", list_3d)
        self._write_drr_required_note(output)
        self._print_export_summary(output)

    def _collect_records(self, source: Path):
        """Read DICOM headers from loose files or ZIP entries without decoding pixels."""
        records = []
        for item in self._iter_sources(source):
            try:
                raw = item["read"]()
                ds = self._read_dicom(raw, stop_before_pixels=True)
                body_part = str(ds.get("BodyPartExamined", "")).upper()
                if "CHEST" not in body_part and "THORAX" not in body_part:
                    self.stats["skipped_non_chest_thorax"] += 1
                    continue
                item["ds"] = ds
                records.append(item)
                self.stats["dicom_records"] += 1
            except Exception as e:
                self.stats["read_failed"] += 1
                self.errors.append({"file": item["name"], "error": str(e)})
        return records

    def _ct_series_skip_reason(self, records):
        """Return why a CT series is not suitable for UniMiSSPlus volume export."""
        descriptions = []
        image_types = []
        for record in records[:5]:
            ds = record["ds"]
            descriptions.append(str(ds.get("SeriesDescription", "")))
            image_types.append(str(ds.get("ImageType", "")))

        text = " ".join(descriptions + image_types).upper()
        # Vietnamese viewers commonly show scout/localizer series as "HINH DINH VI".
        localizer_keywords = (
            "LOCALIZER",
            "SCOUT",
            "TOPOGRAM",
            "SURVIEW",
            "HINH DINH VI",
            "DINH VI",
        )
        if any(keyword in text for keyword in localizer_keywords):
            return "localizer"
        if "SCREEN SAVE" in text or "SCREENSHOT" in text:
            return "screen_save"
        if len(records) < self.min_ct_slices:
            return "too_few_slices"
        return None

    def _iter_sources(self, source: Path):
        """Yield DICOM byte readers for loose files and files inside ZIP archives."""
        paths = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
        for path in paths:
            if path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        names = [n for n in zf.namelist() if not n.endswith("/")]
                    for name in names:
                        yield {
                            "name": f"{path}:{name}",
                            "read": lambda p=path, n=name: zipfile.ZipFile(p, "r").read(n),
                        }
                except Exception as e:
                    self.stats["zip_failed"] += 1
                    self.errors.append({"file": str(path), "error": str(e)})
            else:
                yield {
                    "name": str(path),
                    "read": lambda p=path: p.read_bytes(),
                }

    def _read_dicom(self, raw: bytes, stop_before_pixels: bool = False):
        buffer = BytesIO(raw)
        try:
            return pydicom.dcmread(buffer, force=False, stop_before_pixels=stop_before_pixels)
        except Exception:
            buffer.seek(0)
            return pydicom.dcmread(buffer, force=True, stop_before_pixels=stop_before_pixels)

    def _export_xray_png(self, record, images2d_dir: Path):
        """Convert one 2D X-ray DICOM to the PNG stream consumed by Dataset2D."""
        try:
            ds = self._read_dicom(record["read"]())
            image = self._dicom_to_uint8(ds)
            image = self.Image.fromarray(image).resize((512, 512), self.Image.BILINEAR).convert("RGB")
            name = self._safe_name(ds, record["name"]) + ".png"
            out_path = images2d_dir / name
            image.save(out_path)
            self.stats["xray_png_exported"] += 1
            return f"2D_images/{name}"
        except Exception as e:
            self.stats["xray_failed"] += 1
            self.errors.append({"file": record["name"], "error": str(e)})
            return None

    def _export_ct_series(self, records, images3d_dir: Path):
        """Convert one DICOM CT series into a full NIfTI volume."""
        try:
            slices = []
            for record in records:
                ds = self._read_dicom(record["read"]())
                slices.append((self._slice_sort_key(ds), ds))
            slices.sort(key=lambda item: item[0])

            arrays = []
            for _, ds in slices:
                pixel = ds.pixel_array.astype(self.np.float32)
                slope = float(ds.get("RescaleSlope", 1.0))
                intercept = float(ds.get("RescaleIntercept", 0.0))
                arrays.append(pixel * slope + intercept)

            volume = self.np.stack(arrays, axis=0).astype(self.np.int16)
            volume = self.np.transpose(volume, (1, 2, 0))
            first = slices[0][1]
            affine = self._ct_affine(first, slices)
            name = self._safe_name(first, records[0]["name"]) + ".nii.gz"
            out_path = images3d_dir / name
            self.nib.save(self.nib.Nifti1Image(volume, affine), out_path)
            return out_path
        except Exception as e:
            self.stats["ct_failed"] += 1
            self.errors.append({"file": records[0]["name"], "error": str(e)})
            return None

    def _export_subvolumes(self, nii_path: Path, subvolumes_dir: Path):
        """Cut a full CT NIfTI volume into UniMiSSPlus-style 24-slice chunks."""
        image = self.nib.load(str(nii_path))
        volume = image.get_fdata().astype(self.np.int16)
        depth = volume.shape[2]
        if depth < self.subvolume_depth:
            pad_before = (self.subvolume_depth - depth) // 2
            pad_after = self.subvolume_depth - depth - pad_before
            volume = self.np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), mode="constant")
            depth = volume.shape[2]

        count = int(math.ceil((depth - self.subvolume_depth) / self.subvolume_stride) + 1)
        rel_paths = []
        for dep in range(count):
            d1 = int(dep * self.subvolume_stride)
            d2 = min(d1 + self.subvolume_depth, depth)
            if d2 - d1 < self.subvolume_depth:
                d1 = d2 - self.subvolume_depth

            subvol = volume[:, :, max(d1, 0):d2]
            if subvol.shape[0] > 320:
                subvol = subvol[int(subvol.shape[0] * 0.1):int(subvol.shape[0] * 0.9), int(subvol.shape[1] * 0.1):int(subvol.shape[1] * 0.9), :]

            stem = nii_path.name[:-7] + f"_dep{dep}"
            sub_path = subvolumes_dir / f"{stem}.nii.gz"
            self.nib.save(self.nib.Nifti1Image(subvol.astype(self.np.int16), image.affine), sub_path)
            rel_paths.append(f"3D_subvolumes/{sub_path.name}")
            self.stats["ct_subvolumes_exported"] += 1
        return rel_paths

    def _dicom_to_uint8(self, ds):
        """Window DICOM pixel data to an 8-bit image for X-ray PNG export."""
        arr = ds.pixel_array.astype(self.np.float32)
        slope = float(ds.get("RescaleSlope", 1.0))
        intercept = float(ds.get("RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        return self._window_to_uint8(arr)

    def _window_to_uint8(self, arr):
        finite = arr[self.np.isfinite(arr)]
        if finite.size == 0:
            return self.np.zeros(arr.shape, dtype=self.np.uint8)
        low, high = self.np.percentile(finite, [1, 99])
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            return self.np.zeros(arr.shape, dtype=self.np.uint8)
        arr = self.np.clip((arr - low) / (high - low), 0, 1)
        return (arr * 255).astype(self.np.uint8)

    def _slice_sort_key(self, ds):
        ipp = ds.get("ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
        instance = ds.get("InstanceNumber", None)
        if instance is not None:
            return float(instance)
        return 0.0

    def _ct_affine(self, first, slices):
        """Build a simple spacing-aware affine for exported NIfTI volumes."""
        pixel_spacing = first.get("PixelSpacing", [1.0, 1.0])
        row_spacing = float(pixel_spacing[0])
        col_spacing = float(pixel_spacing[1])
        if len(slices) > 1:
            z_values = [self._slice_sort_key(ds) for _, ds in slices]
            diffs = [abs(b - a) for a, b in zip(z_values[:-1], z_values[1:]) if abs(b - a) > 0]
            slice_spacing = float(self.np.median(diffs)) if diffs else float(first.get("SliceThickness", 1.0))
        else:
            slice_spacing = float(first.get("SliceThickness", 1.0))
        return self.np.diag([row_spacing, col_spacing, slice_spacing, 1.0])

    def _safe_name(self, ds, fallback: str):
        uid = ds.get("SOPInstanceUID", None) or ds.get("SeriesInstanceUID", None) or ds.get("StudyInstanceUID", None) or fallback
        return hashlib.sha256(str(uid).encode()).hexdigest()[:16]

    def _rel_for_list(self, path: Path, root: Path):
        return path.relative_to(root).as_posix()

    def _write_list(self, path: Path, entries):
        path.write_text("".join(f"{entry}\n" for entry in sorted(entries)), encoding="utf-8")

    def _write_drr_required_note(self, output: Path):
        note = (
            "Official DRR generation is required before UniMiSSPlus upstream pretraining.\n\n"
            "The export-unimissplus command creates CT subvolume NIfTI files in 3D_subvolumes/ "
            "and lists them in 3D_images.txt. UniMiSSPlus/data_loader3D2D.py also expects a "
            "paired PNG with the same stem for each listed .nii.gz file.\n\n"
            "Generate those PNG files with the official UniMiSSPlus DRR pipeline in a Linux/CUDA environment:\n\n"
            "cd UniMiSSPlus/pycuda_drr\n"
            "python setup.py install\n"
            "python rendering_DL.py\n\n"
            "Do not start upstream pretraining until every 3D_subvolumes/*.nii.gz has a matching .png.\n"
        )
        (output / "DRR_REQUIRED.txt").write_text(note, encoding="utf-8")

    def _print_export_summary(self, output: Path):
        print("\n" + "="*80)
        print("UNIMISSPLUS EXPORT SUMMARY")
        print("="*80)
        for key, value in sorted(self.stats.items()):
            print(f"  {key.replace('_', ' ').title()}: {value}")
        if self.errors:
            print("\n--- ERRORS (First 10) ---")
            for error in self.errors[:10]:
                print(f"  {error['file']}: {error['error']}")
        print(f"\n  Output data directory: {output}")
        print(f"  Upstream lists: {output / '2D_images.txt'} and {output / '3D_images.txt'}")
        if self.stats.get("ct_subvolumes_exported", 0):
            print(f"  REQUIRED NEXT STEP: run UniMiSSPlus/pycuda_drr/rendering_DL.py to create DRR PNGs.")
            print(f"  See: {output / 'DRR_REQUIRED.txt'}")
        print("="*80)


class UniMiSSPlusVerifier:
    """Validate that exported UniMiSSPlus data is ready for upstream pretraining."""

    def verify(self, data_dir: str):
        root = Path(data_dir)
        if not root.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        required = ["2D_images.txt", "3D_images.txt", "2D_images", "3D_subvolumes"]
        missing_required = [item for item in required if not (root / item).exists()]

        list_2d = self._read_list(root / "2D_images.txt")
        list_3d = self._read_list(root / "3D_images.txt")

        missing_2d = [entry for entry in list_2d if not (root / entry).is_file()]
        missing_3d = [entry for entry in list_3d if not (root / entry).is_file()]
        missing_drr = []
        for entry in list_3d:
            nii_path = root / entry
            if entry.endswith(".nii.gz"):
                png_path = nii_path.with_name(nii_path.name[:-7] + ".png")
                if not png_path.is_file():
                    missing_drr.append(png_path.relative_to(root).as_posix())

        print("\n" + "="*80)
        print("UNIMISSPLUS DATA VERIFY")
        print("="*80)
        print(f"  Data directory: {root}")
        print(f"  2D list entries: {len(list_2d)}")
        print(f"  3D list entries: {len(list_3d)}")
        print(f"  Missing required paths: {len(missing_required)}")
        print(f"  Missing 2D files: {len(missing_2d)}")
        print(f"  Missing 3D NIfTI files: {len(missing_3d)}")
        print(f"  Missing required DRR PNG pairs: {len(missing_drr)}")

        for title, values in [
            ("Missing required paths", missing_required),
            ("Missing 2D files", missing_2d),
            ("Missing 3D NIfTI files", missing_3d),
            ("Missing DRR PNG pairs", missing_drr),
        ]:
            if values:
                print(f"\n--- {title} (first 10) ---")
                for value in values[:10]:
                    print(f"  {value}")

        ready = not missing_required and not missing_2d and not missing_3d and not missing_drr
        if ready:
            print("\n  OK: data is ready for UniMiSSPlus upstream loaders.")
        else:
            print("\n  NOT READY: fix the missing items before upstream pretraining.")
            if missing_drr:
                print("  Run UniMiSSPlus/pycuda_drr/rendering_DL.py to create required DRR PNGs.")
        print("="*80)
        return ready

    def _read_list(self, path: Path):
        if not path.is_file():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description='DICOM Patient Information Analyzer and Anonymizer')
    subparsers = parser.add_subparsers(dest='command', help='Command')

    analyze_parser = subparsers.add_parser('analyze', help='Analyze DICOM files')
    analyze_parser.add_argument('directory', help='Directory or ZIP file')
    analyze_parser.add_argument('--output', '-o', help='Save report to JSON file')
    analyze_parser.add_argument('--no-recursive', action='store_true', help='Do not search recursively')

    anon_parser = subparsers.add_parser('anonymize', help='Anonymize DICOM files')
    anon_parser.add_argument('input_dir', help='Input directory or ZIP file')
    anon_parser.add_argument('output_dir', help='Output directory or ZIP file')

    clean_parser = subparsers.add_parser('clean-invalid-zips', help='Delete .zip files that are not valid ZIP archives')
    clean_parser.add_argument('path', help='Directory or ZIP file to scan')
    clean_parser.add_argument('--dry-run', action='store_true', help='Report invalid ZIP files without deleting them')
    clean_parser.add_argument('--output', '-o', help='Save cleanup report to JSON file')

    export_parser = subparsers.add_parser('export-unimissplus', help='Export DICOM data into UniMiSSPlus upstream data format')
    export_parser.add_argument('input', help='Input directory or ZIP file containing DICOM files')
    export_parser.add_argument('output_data_dir', help='Output UniMiSSPlus data directory, e.g. UniMiSSPlus/data')
    export_parser.add_argument('--subvolume-depth', type=int, default=24, help='Depth of exported CT subvolumes')
    export_parser.add_argument('--subvolume-stride', type=int, default=12, help='Stride between exported CT subvolumes')
    export_parser.add_argument('--min-ct-slices', type=int, default=24, help='Skip CT series with fewer slices than this')

    verify_parser = subparsers.add_parser('verify-unimissplus', help='Verify UniMiSSPlus exported data and required DRR PNG pairs')
    verify_parser.add_argument('data_dir', help='UniMiSSPlus data directory to verify, e.g. UniMiSSPlus/data')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'analyze':
        analyzer = DICOMAnalyzer()
        report = analyzer.analyze_directory(args.directory, not args.no_recursive)
        analyzer.print_report(report)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.output}")

    elif args.command == 'anonymize':
        DICOMAnonymizer().anonymize_directory(args.input_dir, args.output_dir)

    elif args.command == 'clean-invalid-zips':
        report = InvalidZipCleaner(dry_run=args.dry_run).clean(args.path)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"\nCleanup report saved to: {args.output}")

    elif args.command == 'export-unimissplus':
        exporter = UniMiSSPlusExporter(
            subvolume_depth=args.subvolume_depth,
            subvolume_stride=args.subvolume_stride,
            min_ct_slices=args.min_ct_slices,
        )
        try:
            exporter.export(args.input, args.output_data_dir)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == 'verify-unimissplus':
        ready = UniMiSSPlusVerifier().verify(args.data_dir)
        if not ready:
            sys.exit(1)


if __name__ == '__main__':
    main()
