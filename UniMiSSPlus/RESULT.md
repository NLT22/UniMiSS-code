# UniMiSS+ Fine-tuning Results

This file summarizes the experiment artifacts saved under `results/`.

## Overview

| Experiment | Task | Split / Setting | Primary metric | AUC | AP | Notes |
|---|---:|---|---:|---:|---:|---|
| COVID-19 Radiography | 3-class | Test on COVID-19 Radiography | Accuracy `0.9657` | Macro AUC `0.9960` | Macro AP `0.9934` | Eval from `results/covid_other_normal/best.pth` |
| COVID-QU-Ex | 3-class | Test on COVID-QU-Ex | Accuracy `0.9586` | Macro AUC `0.9950` | Macro AP `0.9900` | Eval from `results/train_quex/best.pth` |
| QU-Ex to COVID-19 | 3-class | Cross-dataset eval | Accuracy `0.9612` | Macro AUC `0.9962` | Macro AP `0.9941` | QU-Ex checkpoint evaluated on COVID-19 Radiography |
| COVID-19 to QU-Ex | 3-class | Cross-dataset eval | Accuracy `0.9443` | Macro AUC `0.9942` | Macro AP `0.9884` | COVID-19 checkpoint evaluated on COVID-QU-Ex |
| NIH ChestX-ray14 | Multi-label | NIH test list | Mean AUC `0.7988` | Mean AUC `0.7988` | Mean AP `0.2916` | Eval from `results/unimiss_nih/best.pth` |

All training summaries currently contain seed `42`.

## Training Summary

| Experiment | Best seed | Best epoch | Train loss | Best score |
|---|---:|---:|---:|---:|
| COVID-19 Radiography | `42` | `31` | `0.1606` | Accuracy `0.9511` |
| COVID-QU-Ex | `42` | `39` | `0.1768` | Accuracy `0.9586` |
| NIH ChestX-ray14 | `42` | `29` | `0.1595` | Mean AUC `0.7988` |

### Training Curves

| COVID-19 Radiography | COVID-QU-Ex | NIH ChestX-ray14 |
|---|---|---|
| ![](results/covid_other_normal/seed_42/training_curves.png) | ![](results/train_quex/seed_42/training_curves.png) | ![](results/unimiss_nih/seed_42/training_curves.png) |

## COVID-19 Radiography Evaluation

Metrics from `results/covid_other_normal_eval/eval_metrics.json`.

| Accuracy | Macro AUC | Macro AP |
|---:|---:|---:|
| `0.9657` | `0.9960` | `0.9934` |

Classification report summary:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| COVID | `0.97` | `0.98` | `0.97` | `723` |
| Other | `0.95` | `0.96` | `0.96` | `1471` |
| Normal | `0.97` | `0.96` | `0.97` | `2038` |

### Plots

| Confusion Matrix | ROC Curves | PR Curves |
|---|---|---|
| ![](results/covid_other_normal_eval/eval_confusion_matrix.png) | ![](results/covid_other_normal_eval/eval_roc_curves.png) | ![](results/covid_other_normal_eval/eval_pr_curves.png) |

### Grad-CAM

| COVID | Other | Normal |
|---|---|---|
| ![](results/covid_other_normal_eval/grad_cam/grad_cam_true_00_COVID_00.png) | ![](results/covid_other_normal_eval/grad_cam/grad_cam_true_01_Other_00.png) | ![](results/covid_other_normal_eval/grad_cam/grad_cam_true_02_Normal_00.png) |

## COVID-QU-Ex Evaluation

Metrics from `results/train_quex_eval/eval_metrics.json`.

| Accuracy | Macro AUC | Macro AP |
|---:|---:|---:|
| `0.9586` | `0.9950` | `0.9900` |

Classification report summary:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| COVID-19 | `0.99` | `0.97` | `0.98` | `2395` |
| Non-COVID | `0.93` | `0.97` | `0.95` | `2253` |
| Normal | `0.95` | `0.94` | `0.95` | `2140` |

### Plots

| Confusion Matrix | ROC Curves | PR Curves |
|---|---|---|
| ![](results/train_quex_eval/eval_confusion_matrix.png) | ![](results/train_quex_eval/eval_roc_curves.png) | ![](results/train_quex_eval/eval_pr_curves.png) |

### Grad-CAM

| COVID-19 | Non-COVID | Normal |
|---|---|---|
| ![](results/train_quex_eval/grad_cam/grad_cam_true_00_COVID-19_00.png) | ![](results/train_quex_eval/grad_cam/grad_cam_true_01_Non-COVID_00.png) | ![](results/train_quex_eval/grad_cam/grad_cam_true_02_Normal_00.png) |

## Cross-Dataset Evaluation: QU-Ex Checkpoint on COVID-19

Metrics from `results/eval_quex_to_covid19/eval_metrics.json`.

| Accuracy | Macro AUC | Macro AP |
|---:|---:|---:|
| `0.9612` | `0.9962` | `0.9941` |

Classification report summary:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| COVID | `0.97` | `0.98` | `0.98` | `723` |
| Other | `0.94` | `0.97` | `0.95` | `1471` |
| Normal | `0.98` | `0.95` | `0.96` | `2038` |

### Plots

| Confusion Matrix | ROC Curves | PR Curves |
|---|---|---|
| ![](results/eval_quex_to_covid19/eval_confusion_matrix.png) | ![](results/eval_quex_to_covid19/eval_roc_curves.png) | ![](results/eval_quex_to_covid19/eval_pr_curves.png) |

## Cross-Dataset Evaluation: COVID-19 Checkpoint on QU-Ex

Metrics from `results/eval_covid19_to_quex/eval_metrics.json`.

| Accuracy | Macro AUC | Macro AP |
|---:|---:|---:|
| `0.9443` | `0.9942` | `0.9884` |

Classification report summary:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| COVID-19 | `0.99` | `0.90` | `0.94` | `2395` |
| Non-COVID | `0.93` | `0.97` | `0.95` | `2253` |
| Normal | `0.92` | `0.97` | `0.94` | `2140` |

### Plots

| Confusion Matrix | ROC Curves | PR Curves |
|---|---|---|
| ![](results/eval_covid19_to_quex/eval_confusion_matrix.png) | ![](results/eval_covid19_to_quex/eval_roc_curves.png) | ![](results/eval_covid19_to_quex/eval_pr_curves.png) |

## NIH ChestX-ray14 Evaluation

Metrics from `results/unimiss_nih_eval/eval_metrics.json`.

| Mean AUC | Mean AP | Micro F1 | Exact Match Accuracy | Label Accuracy |
|---:|---:|---:|---:|---:|
| `0.7988` | `0.2916` | `0.3492` | `0.2484` | `0.9118` |

### Per-Class AUC/AP

| Class | AUC | AP |
|---|---:|---:|
| Atelectasis | `0.7707` | `0.3378` |
| Cardiomegaly | `0.8813` | `0.3380` |
| Effusion | `0.8260` | `0.5112` |
| Infiltration | `0.7091` | `0.4102` |
| Mass | `0.8149` | `0.3151` |
| Nodule | `0.7645` | `0.2265` |
| Pneumonia | `0.7186` | `0.0519` |
| Pneumothorax | `0.8562` | `0.4241` |
| Consolidation | `0.7473` | `0.1600` |
| Edema | `0.8451` | `0.1697` |
| Emphysema | `0.8706` | `0.3406` |
| Fibrosis | `0.8031` | `0.1007` |
| Pleural_Thickening | `0.7615` | `0.1282` |
| Hernia | `0.8772` | `0.1976` |
| No Finding | `0.7357` | `0.6631` |

### Plots

| Micro ROC | Micro PR | Label Frequency |
|---|---|---|
| ![](results/unimiss_nih_eval/eval_micro_roc.png) | ![](results/unimiss_nih_eval/eval_micro_pr.png) | ![](results/unimiss_nih_eval/eval_label_frequency.png) |

| Per-Class AUC | Per-Class AP |
|---|---|
| ![](results/unimiss_nih_eval/eval_per_class_auc.png) | ![](results/unimiss_nih_eval/eval_per_class_ap.png) |

### Grad-CAM

NIH is a multi-label task, so Grad-CAM is saved for predicted target classes instead of one balanced image per true class.

| Sample 0 | Sample 1 | Sample 2 | Sample 3 |
|---|---|---|---|
| ![](results/unimiss_nih_eval/grad_cam/grad_cam_0000_class_14.png) | ![](results/unimiss_nih_eval/grad_cam/grad_cam_0001_class_14.png) | ![](results/unimiss_nih_eval/grad_cam/grad_cam_0002_class_14.png) | ![](results/unimiss_nih_eval/grad_cam/grad_cam_0003_class_14.png) |

| Sample 4 | Sample 5 | Sample 6 | Sample 7 |
|---|---|---|---|
| ![](results/unimiss_nih_eval/grad_cam/grad_cam_0004_class_14.png) | ![](results/unimiss_nih_eval/grad_cam/grad_cam_0005_class_14.png) | ![](results/unimiss_nih_eval/grad_cam/grad_cam_0006_class_14.png) | ![](results/unimiss_nih_eval/grad_cam/grad_cam_0007_class_14.png) |

## ONNX Export And Inference Benchmark

All listed ONNX benchmarks used `CUDAExecutionProvider` with `CPUExecutionProvider` fallback available.

### Speed

| Experiment | PyTorch img/s | ONNX img/s | Throughput delta | PyTorch ms/img | ONNX ms/img | Latency delta |
|---|---:|---:|---:|---:|---:|---:|
| COVID-19 Radiography | `637.18` | `594.97` | `-6.62%` | `1.5694` | `1.6808` | `+7.09%` |
| COVID-QU-Ex | `635.28` | `592.69` | `-6.70%` | `1.5741` | `1.6872` | `+7.19%` |
| NIH ChestX-ray14 | `635.62` | `561.00` | `-11.74%` | `1.5733` | `1.7825` | `+13.30%` |

ONNX CUDA preserved predictions, but it was slower than PyTorch CUDA in these runs.

### Accuracy And Metric Parity

| Experiment | Metric | PyTorch | ONNX | Delta | Assessment |
|---|---|---:|---:|---:|---|
| COVID-19 Radiography | Accuracy | `0.965737` | `0.965737` | `+0.000000` | No accuracy drop |
| COVID-19 Radiography | Macro AUC | `0.995976` | `0.995976` | `-0.000000` | Negligible |
| COVID-19 Radiography | Macro AP | `0.993422` | `0.993421` | `-0.000000` | Negligible |
| COVID-QU-Ex | Accuracy | `0.958603` | `0.958603` | `+0.000000` | No accuracy drop |
| COVID-QU-Ex | Macro AUC | `0.995027` | `0.995027` | `-0.000000` | Negligible |
| COVID-QU-Ex | Macro AP | `0.989962` | `0.989962` | `+0.000000` | No drop |
| NIH ChestX-ray14 | Mean AUC | `0.798795` | `0.798796` | `+0.000001` | No drop |
| NIH ChestX-ray14 | Mean AP | `0.291649` | `0.291649` | `+0.000000` | No drop |
| NIH ChestX-ray14 | Micro F1 | `0.349225` | `0.349264` | `+0.000038` | No drop |
| NIH ChestX-ray14 | Exact Match Accuracy | `0.248359` | `0.248398` | `+0.000039` | No drop |
| NIH ChestX-ray14 | Label Accuracy | `0.911830` | `0.911835` | `+0.000005` | No drop |

### Output Difference And Prediction Agreement

| Experiment | Max abs diff | Mean abs diff | Prediction agreement |
|---|---:|---:|---:|
| COVID-19 Radiography | `0.000569` | `0.000011` | `1.000000` |
| COVID-QU-Ex | `0.000736` | `0.000009` | `1.000000` |
| NIH ChestX-ray14 | `0.000340` | `0.000012` | label `0.999995`, exact `0.999922` |

Conclusion: ONNX export did not cause a meaningful accuracy or metric regression. The only observed regression is runtime speed: ONNX Runtime CUDA is slower than PyTorch CUDA for these benchmark settings.

## Artifact Index

| Experiment | Metrics | Report | Benchmark | Grad-CAM Summary |
|---|---|---|---|---|
| COVID-19 Radiography | [eval_metrics.json](results/covid_other_normal_eval/eval_metrics.json) | [eval_classification_report.txt](results/covid_other_normal_eval/eval_classification_report.txt) | [inference_benchmark.json](results/covid_other_normal_eval/inference_benchmark.json) | [grad_cam_summary.json](results/covid_other_normal_eval/grad_cam/grad_cam_summary.json) |
| COVID-QU-Ex | [eval_metrics.json](results/train_quex_eval/eval_metrics.json) | [eval_classification_report.txt](results/train_quex_eval/eval_classification_report.txt) | [inference_benchmark.json](results/train_quex_eval/inference_benchmark.json) | [grad_cam_summary.json](results/train_quex_eval/grad_cam/grad_cam_summary.json) |
| QU-Ex to COVID-19 | [eval_metrics.json](results/eval_quex_to_covid19/eval_metrics.json) | [eval_classification_report.txt](results/eval_quex_to_covid19/eval_classification_report.txt) | N/A | N/A |
| COVID-19 to QU-Ex | [eval_metrics.json](results/eval_covid19_to_quex/eval_metrics.json) | [eval_classification_report.txt](results/eval_covid19_to_quex/eval_classification_report.txt) | N/A | N/A |
| NIH ChestX-ray14 | [eval_metrics.json](results/unimiss_nih_eval/eval_metrics.json) | [eval_classification_report.txt](results/unimiss_nih_eval/eval_classification_report.txt) | [inference_benchmark.json](results/unimiss_nih_eval/inference_benchmark.json) | [grad_cam_summary.json](results/unimiss_nih_eval/grad_cam/grad_cam_summary.json) |
