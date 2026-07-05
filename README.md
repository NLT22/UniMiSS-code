# Phát hiện bất thường trên ảnh X-quang ngực bằng UniMiSS+

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `Script/` | Pipeline xử lý DICOM: phân tích, ẩn danh, xuất ảnh, gán nhãn, dựng fold, các thí nghiệm |
| `Script/dicom_analyzer.py` | Quét/ẩn danh DICOM, xuất PNG 512×512 sang định dạng UniMiSS+ |
| `Script/dicom_labeler.py` | Kết luận chẩn đoán → nhãn nhị phân, dựng danh sách CV gộp theo bệnh nhân |
| `Script/make_drr.py`, `build_drr_augmented_folds.py` | Dựng ảnh X-quang mô phỏng (DRR) từ CT bằng DiffDRR, thêm vào fold huấn luyện |
| `Script/cross_dataset_eval.py` | Đánh giá chéo zero-shot mô hình NIH/COVID lên dữ liệu của ta |
| `Script/summarize_unimiss_cv.py`, `bootstrap_ci.py` | Tổng hợp CV, gộp dự đoán ngoài-phần, khoảng tin cậy bootstrap |
| `Script/results/` | Kết quả thí nghiệm (chỉ giữ bộ chính thức 1108 ảnh; checkpoint `*.pth`/`*.npz` bị gitignore) |
| `UniMiSSPlus/` | Mã UniMiSS+ (TPAMI) gốc; `Downstream/2D/Cls/main_flexible.py` là entry huấn luyện/đánh giá |
| `UniMiSSPlus/run_vietnam_xray_cv.sh` | Script chạy 5-fold CV cho bài toán Normal/Abnormal |
| `latex/` | Báo cáo LaTeX |
| `UniMissPlus.pth` | Trọng số tiền huấn luyện UniMiSS+ (gitignore, ~1 GB) |

## Môi trường

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Thí nghiệm chạy trên 1 GPU (RTX 5060 Ti 16 GB). PyTorch trong `requirements.txt` là bản
CUDA 12.8 — xem chú thích trong file để cài đúng wheel cho GPU/CUDA của bạn. Trọng số
`UniMissPlus.pth` đặt ở gốc repo.

## Dữ liệu (không kèm trong repo)

Dữ liệu bệnh nhân bị gitignore. Cần có sẵn:
- `Script/ANONYMIZE/` — DICOM đã ẩn danh + `LABELS_FINAL.xlsx`
- `Script/UniMiSSPlus_data/2D_images/` — ảnh X-quang PNG 512×512 đã xuất
- `Script/labels.xlsx` (nhãn — xem dưới), `Script/ANONYMIZE_meta/patient_study_map.csv`
- 11 ảnh có chú thích CAD in đè đã được tách sang `Script/results/quarantine_overlay/`
  (nên tập phân loại là **1.108 ảnh**: 804 Normal / 304 Abnormal).

### Nhãn — một file Excel duy nhất

**`Script/labels.xlsx`** là file nhãn **duy nhất** của toàn pipeline: mỗi dòng một lượt chụp
(1.208 lượt, 791 Normal / 417 Abnormal). `dicom_labeler.py` đọc/ghi được cả `.xlsx` lẫn `.csv`
theo đuôi file, nên mọi bước (gán nhãn → dựng fold → bảng thống kê) đều chạy thẳng từ file này —
không còn `labels_raw.csv`/`labels_classified.csv` trung gian.

Các cột chính:
| Cột | Ý nghĩa |
|---|---|
| `label_match_id`, `modality` | mã lượt chụp (băm) + phương thức (X-ray/CT) |
| `coarse_label`, `normal_abnormal_class` | Normal/Abnormal — **`normal_abnormal_class`: 0 = Abnormal (lớp dương), 1 = Normal** (có chú thích trong ô) |
| `disease_label`, `multi_labels` | nhóm bệnh suy ra từ kết luận |
| `evidence`, `label_reason` | cụm từ/luật đã khớp để ra nhãn |
| `conclusion` | kết luận gốc của bác sĩ |

Nhãn mức **ảnh** cho model là *output* sinh ra ở `Script/labels/vietnam_xray_cv_clean/fold_*/*.txt`
(định dạng `2D_images/<hash>.png <lop>`, cùng quy ước 0=Abnormal / 1=Normal).

---

## Tái lập kết quả

Toàn bộ số liệu trong báo cáo được sinh trên bộ dữ liệu sạch **1.108 ảnh**, huấn luyện
**batch size 32, 30 epoch, learning rate 1e-4, AdamW**, 5-fold CV gộp theo bệnh nhân.

### 0. Chuẩn bị nhãn và fold (chạy một lần)

```bash
# (chỉ khi tạo nhãn từ đầu) trích xuất kết luận rồi gán nhãn -> MỘT file Excel duy nhất.
# labels_raw.csv là trung gian tạm; nhãn dùng abnormal-priority (cụm bất thường kiểm trước).
python Script/dicom_labeler.py extract Script/ANONYMIZE/LABELS_FINAL.xlsx Script/ANONYMIZE --output /tmp/labels_raw.csv
python Script/dicom_labeler.py classify /tmp/labels_raw.csv --output Script/labels.xlsx

# Dựng 5-fold CV gộp theo bệnh nhân — ĐỌC THẲNG Script/labels.xlsx
# (11 ảnh overlay đã tách nên tự loại -> 1108 ảnh)
python Script/dicom_labeler.py build-cv-lists \
    Script/labels.xlsx Script/ANONYMIZE Script/UniMiSSPlus_data \
    --patient-map Script/ANONYMIZE_meta/patient_study_map.csv \
    --output-dir Script/labels/vietnam_xray_cv_clean

# Sinh hai bảng nhóm triệu chứng trong báo cáo (bang:rules + bang:findings) từ
# cùng một taxonomy -> luôn nhất quán với nhau và tái lập được:
python Script/build_findings_table.py            # in ra body LaTeX của cả hai bảng
```

### 1. Thí nghiệm chính (Bảng "tiền huấn luyện" + độ ổn định)

Chạy qua `run_vietnam_xray_cv.sh` (điều khiển bằng biến môi trường). Luôn đặt `BATCH_SIZE=32`.

```bash
cd UniMiSSPlus
CLEAN=../Script/labels/vietnam_xray_cv_clean

# Tinh chỉnh (baseline chính, cũng là baseline cho thí nghiệm DRR)
CV_DIR=$CLEAN OUT_ROOT=../Script/results/unimiss_vietnam_xray_clean \
  SEED=42 EPOCHS=30 BATCH_SIZE=32 ./run_vietnam_xray_cv.sh

# Huấn luyện từ đầu (đối chứng, khởi tạo ngẫu nhiên)
CV_DIR=$CLEAN OUT_ROOT=../Script/results/unimiss_vietnam_xray_clean_scratch \
  NO_PRETRAIN=1 SEED=42 EPOCHS=30 BATCH_SIZE=32 ./run_vietnam_xray_cv.sh

# Độ ổn định đa seed
CV_DIR=$CLEAN OUT_ROOT=../Script/results/unimiss_vietnam_xray_clean_seed123  SEED=123  EPOCHS=30 BATCH_SIZE=32 ./run_vietnam_xray_cv.sh
CV_DIR=$CLEAN OUT_ROOT=../Script/results/unimiss_vietnam_xray_clean_seed2024 SEED=2024 EPOCHS=30 BATCH_SIZE=32 ./run_vietnam_xray_cv.sh

# Hàm mất mát focal (ASL, gamma_pos=2)
CV_DIR=$CLEAN OUT_ROOT=../Script/results/unimiss_vietnam_xray_clean_focal \
  SEED=42 EPOCHS=30 BATCH_SIZE=32 LOSS=asl ASL_GAMMA_NEG=4 ASL_GAMMA_POS=2 ./run_vietnam_xray_cv.sh
```

### 2. Tăng cường dữ liệu bằng ảnh mô phỏng từ CT (Bảng DRR)

```bash
# Dựng DRR từ các khối CT (Siddon, cắt theo hộp bao cơ thể)
python Script/make_drr.py --out-dir Script/UniMiSSPlus_data/2D_images_drr

# Thêm DRR vào phần huấn luyện của mỗi fold (loại trừ theo bệnh nhân)
python Script/build_drr_augmented_folds.py \
    --cv-dir Script/labels/vietnam_xray_cv_clean \
    --drr-dir Script/UniMiSSPlus_data/2D_images_drr \
    --labels Script/labels.xlsx \
    --patient-map Script/ANONYMIZE_meta/patient_study_map.csv \
    --out-dir Script/labels/vietnam_xray_cv_clean_drr --abnormal-repeat 3

cd UniMiSSPlus
CV_DIR=../Script/labels/vietnam_xray_cv_clean_drr \
  OUT_ROOT=../Script/results/unimiss_vietnam_xray_clean_drr \
  SEED=42 EPOCHS=30 BATCH_SIZE=32 ./run_vietnam_xray_cv.sh
```

### 3. Đánh giá chéo với NIH / COVID (Bảng cross-dataset)

Dùng checkpoint đã huấn luyện sẵn của dự án trước (đường dẫn `--old-results`, không train lại):

```bash
python Script/cross_dataset_eval.py \
    --old-results <đường/dẫn/tới/UniMiSSPlus/results> \
    --data-root Script/UniMiSSPlus_data \
    --manifest Script/labels/vietnam_xray_cv_clean/cv_manifest.csv \
    --output Script/results/cross_dataset_eval_clean.csv
```

### 4. Tổng hợp số liệu + khoảng tin cậy

```bash
# Trung bình theo fold + gộp dự đoán ngoài-phần (mỗi thí nghiệm)
python Script/summarize_unimiss_cv.py \
    --results-root Script/results/unimiss_vietnam_xray_clean --folds 5 \
    --output-dir Script/results/unimiss_vietnam_xray_clean/summary

# Bootstrap 95% CI trên dự đoán gộp
python Script/bootstrap_ci.py
```

### 5. Biên dịch báo cáo

```bash
cd latex && latexmk -pdf -interaction=nonstopmode main.tex
```

---

## Kết quả chính (1.108 ảnh)

| Thí nghiệm | AUROC | AP | Recall (Abn) | Specificity | Bal. Acc |
|---|---|---|---|---|---|
| UniMiSS+ tinh chỉnh | 0,833 | 0,634 | 0,737 | 0,786 | 0,762 |
| Huấn luyện từ đầu | 0,810 | 0,591 | 0,639 | 0,796 | 0,717 |
| Focal (ASL) | 0,843 | 0,673 | 0,749 | 0,805 | 0,777 |
| + ảnh mô phỏng CT | 0,838 | — | 0,693 | 0,829 | 0,761 |
| NIH zero-shot | 0,848 | 0,747 | 0,417 | 0,968 | — |
| COVID zero-shot | 0,774 | 0,608 | 0,770 | 0,613 | — |

Đa seed (42/123/2024): AUROC 0,833 / 0,845 / 0,828, độ lệch chuẩn ≈ 0,007;
gộp 15 lần chạy: AUROC 0,825, bootstrap 95% CI [0,809 – 0,840].

**Kết luận cốt lõi:** ở quy mô ~1.100 ảnh, tiền huấn luyện có lợi (so với huấn luyện từ
đầu), nhưng lựa chọn nguồn/hàm mất mát/lấy mẫu chỉ dịch điểm hoạt động chứ không nâng
khả năng phân biệt; một mô hình NIH áp thẳng (không tinh chỉnh) đạt AUROC ngang bằng —
cho thấy **quy mô dữ liệu nguồn quan trọng hơn tinh chỉnh theo miền** khi dữ liệu đích nhỏ.
