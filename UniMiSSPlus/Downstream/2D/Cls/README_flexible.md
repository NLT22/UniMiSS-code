# Flexible UniMiSS+ 2D Classification

`main_flexible.py` adds path-driven downstream training/evaluation without changing the original downstream entrypoint.

Supported tasks:

- `covid`: COVID-19 Radiography, 3-class single-label classification.
- `covid_qu_ex`: COVID-QU-Ex, 3-class single-label classification.
- `nih`: NIH ChestX-ray14, 15-output multi-label classification: 14 diseases plus `No Finding`.

## Quick Start

Common training flags:

```bash
--output_dir "/kaggle/working/run_name" \
--pre_train \
--pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
--batch_size 32 \
--test_batch_size 32 \
--num_workers 4 \
--epochs 30 \
--learning_rate 0.0001
```

On Kaggle, `--num_workers 4` is a good starting point. If the notebook hangs or DataLoader warns about workers, use `--num_workers 2`.

Default learning-rate schedule:

```bash
--lr_schedule poly --lr_power 0.9
```

Other choices:

```bash
--lr_schedule constant
--lr_schedule cosine --min_lr 0.000001
```

## Tasks

### COVID-19 Radiography

Expected structure:

```text
COVID-19_Radiography_Dataset/
  COVID/images/*.png
  Lung_Opacity/images/*.png
  Normal/images/*.png
  Viral Pneumonia/images/*.png
```

Default labels are `COVID,Lung_Opacity,Normal`; `Viral Pneumonia` is ignored unless included in `--covid_classes`.

Recommended 3-class setup for cross-dataset experiments:

```text
COVID
Other = Lung_Opacity + Viral Pneumonia
Normal
```

Use:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/train_covid19" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

Optional fixed split lists are supported with `--covid_train_list` and `--covid_test_list`:

```text
COVID/images/COVID-1.png 0
Lung_Opacity/images/Lung_Opacity-1.png 1
Normal/images/Normal-1.png 2
```

### COVID-QU-Ex

Expected structure:

```text
COVID-QU-Ex/
  Lung Segmentation Data/Lung Segmentation Data/
    Train/COVID-19/images/*.png
    Train/Non-COVID/images/*.png
    Train/Normal/images/*.png
    Val/COVID-19/images/*.png
    Val/Non-COVID/images/*.png
    Val/Normal/images/*.png
    Test/COVID-19/images/*.png
    Test/Non-COVID/images/*.png
    Test/Normal/images/*.png
```

Labels are `COVID-19,Non-COVID,Normal`. By default, training uses `Train,Val` and testing uses `Test`.

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid_qu_ex \
  --covid_qu_ex_root "/kaggle/input/covid-qu-ex/COVID-QU-Ex" \
  --output_dir "/kaggle/working/train_quex" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

Notes:

- If Kaggle points directly to the nested folder, pass that path to `--covid_qu_ex_root`.
- The infection segmentation copy has the same class layout and can be selected with `--covid_qu_ex_subset infection`.

### NIH ChestX-ray14

Expected structure:

```text
NIH CXR/
  Data_Entry_2017.csv
  train_val_list.txt
  test_list.txt
  images_001/images/*.png
  ...
  images_012/images/*.png
```

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task nih \
  --nih_root "/kaggle/input/nih-chest-xrays/data" \
  --nih_csv "/kaggle/input/nih-chest-xrays/data/Data_Entry_2017.csv" \
  --nih_train_list "/kaggle/input/nih-chest-xrays/data/train_val_list.txt" \
  --nih_test_list "/kaggle/input/nih-chest-xrays/data/test_list.txt" \
  --output_dir "/kaggle/working/train_nih" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

NIH is multi-label. The script reports `mean_auc`, `mean_ap`, `micro_f1`, `exact_match_accuracy`, and `label_accuracy`.

## Outputs

Training outputs:

- `last.pth`: final epoch weights.
- `best.pth`: best score weights.
- `checkpoint.pth`: full resume checkpoint with model, optimizer, epoch, and best score.
- `best_metrics.json`: metrics from the best epoch.
- `metrics.jsonl`: one metrics row per epoch.
- `training_curves.png`: training loss, learning rate, and validation/test curves.

Evaluation artifacts are saved by default. Use `--no_plots` to disable PNG/TXT/NPZ artifacts. Training-time eval files use `latest_eval_`; `--eval_only` files use `eval_`.

For `covid` and `covid_qu_ex`, artifacts include confusion matrix, ROC/PR curves, classification report, metrics JSON, and predictions NPZ.

For `nih`, artifacts include per-class AUC/AP plots, label frequency, micro ROC/PR curves, classification report, metrics JSON, and predictions NPZ.

## Evaluation and Resume

Evaluate a saved model:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/eval_covid19" \
  --checkpoint_path "/kaggle/working/train_covid19/best.pth" \
  --eval_only
```

Resume training from full checkpoint:

```bash
--resume_path "/kaggle/working/train_covid19/checkpoint.pth"
```

`--epochs` is the final target epoch count, not the number of extra epochs. Use `--checkpoint_path` for plain weights such as `best.pth` or `last.pth`; use `--resume_path` when restoring optimizer state and schedule position.

## ONNX and Inference Benchmark

Export ONNX after training or eval:

```bash
--export_onnx
```

Defaults:

- Training exports `best.onnx` from `best.pth`.
- `--eval_only` exports `eval.onnx` from the loaded model.
- ONNX input: `image`, shape `N,3,input_size,input_size`.
- ONNX output: `logits`.

Customize:

```bash
--onnx_path "/kaggle/working/unimiss_covid/best.onnx" \
--onnx_opset 17 \
--onnx_export_weights best
```

`--onnx_export_weights` can be `best`, `last`, or `current`.

Compare PyTorch and ONNX Runtime accuracy/speed:

```bash
pip install onnx onnxruntime
```

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/eval_covid19" \
  --checkpoint_path "/kaggle/working/train_covid19/best.pth" \
  --eval_only \
  --export_onnx \
  --compare_onnx \
  --benchmark_warmup_batches 2
```

This writes `inference_benchmark.json` with PyTorch metrics/speed, ONNX metrics/speed, output differences, prediction agreement, and accuracy delta.

Useful benchmark flags:

```bash
--benchmark_inference
--benchmark_max_batches 20
--onnx_runtime_provider auto
```

## Grad-CAM

Save Grad-CAM overlays for test samples:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/eval_covid19" \
  --checkpoint_path "/kaggle/working/train_covid19/best.pth" \
  --eval_only \
  --grad_cam \
  --grad_cam_samples 8
```

Outputs are saved to `<output_dir>/grad_cam` with a `grad_cam_summary.json`. By default, the explained class is the model prediction for each sample. Useful options:

```bash
--grad_cam_class 0
--grad_cam_layer patch_embed2D4.proj1.conv
--grad_cam_weights best
--grad_cam_dir "/kaggle/working/grad_cam"
```

After training, Grad-CAM uses `best.pth` by default. Use `--grad_cam_weights last` or `--grad_cam_weights current` to change that.

## Multi-GPU

Use DataParallel with multiple visible GPUs:

```bash
--gpu "0,1" --multi_gpu --batch_size 64 --test_batch_size 64
```

Saved `last.pth`, `best.pth`, and `checkpoint.pth` do not contain the `module.` prefix, so they can be loaded on single-GPU or multi-GPU runs.

## Cross-Dataset Evaluation

Use aligned COVID label spaces:

```text
COVID-19 Radiography: COVID, Other, Normal
COVID-QU-Ex:          COVID-19, Non-COVID, Normal
```

Alignment:

```text
COVID  -> COVID-19
Other  -> Non-COVID
Normal -> Normal
```

Train on COVID-19 Radiography, evaluate on COVID-QU-Ex:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid_qu_ex \
  --covid_qu_ex_root "/kaggle/input/covid-qu-ex/COVID-QU-Ex" \
  --output_dir "/kaggle/working/eval_covid19_to_quex" \
  --checkpoint_path "/kaggle/working/train_covid19/best.pth" \
  --eval_only
```

Train on COVID-QU-Ex, evaluate on COVID-19 Radiography:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/eval_quex_to_covid19" \
  --checkpoint_path "/kaggle/working/train_quex/best.pth" \
  --eval_only
```

Avoid evaluating a model trained with default COVID labels `COVID,Lung_Opacity,Normal` directly on COVID-QU-Ex, because `Lung_Opacity` and `Non-COVID` are not the same label definition.
