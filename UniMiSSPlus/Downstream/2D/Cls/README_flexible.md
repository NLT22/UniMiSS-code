# Flexible UniMiSS+ 2D Classification

This entrypoint keeps the original downstream code untouched and adds path-driven training for:

- COVID-19 Radiography: 3-class single-label classification.
- COVID-QU-Ex: 3-class single-label classification.
- NIH ChestX-ray14: 14 disease labels plus `No Finding` as a 15-output multi-label task.

## COVID-19 Radiography

Expected structure:

```text
COVID-19_Radiography_Dataset/
  COVID/images/*.png
  Lung_Opacity/images/*.png
  Normal/images/*.png
  Viral Pneumonia/images/*.png
```

Default COVID classes are:

```text
COVID,Lung_Opacity,Normal
```

`Viral Pneumonia` is ignored unless passed in `--covid_classes`.

Example:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/unimiss_covid" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

To use the requested 3-class grouping `COVID`, `Other`, `Normal`, where `Other = Lung_Opacity + Viral Pneumonia`, add:

```bash
--covid_mode other_normal
```

Full example:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/unimiss_covid_other_normal" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

Optional fixed split lists are supported:

```text
COVID/images/COVID-1.png 0
Lung_Opacity/images/Lung_Opacity-1.png 1
Normal/images/Normal-1.png 2
```

Pass them with `--covid_train_list` and `--covid_test_list`.

## COVID-QU-Ex

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

The labels are:

```text
COVID-19,Non-COVID,Normal
```

By default, training uses `Train,Val` and testing uses `Test`.

Example:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid_qu_ex \
  --covid_qu_ex_root "/kaggle/input/covid-qu-ex/COVID-QU-Ex" \
  --output_dir "/kaggle/working/unimiss_covid_qu_ex" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

If the Kaggle input points directly to the nested folder, this also works:

```bash
--covid_qu_ex_root "/kaggle/input/covid-qu-ex/Lung Segmentation Data/Lung Segmentation Data"
```

The infection segmentation copy has the same class folder layout and can be selected with:

```bash
--covid_qu_ex_subset infection
```

## NIH ChestX-ray14

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

Example:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task nih \
  --nih_root "/kaggle/input/nih-chest-xrays/data" \
  --nih_csv "/kaggle/input/nih-chest-xrays/data/Data_Entry_2017.csv" \
  --nih_train_list "/kaggle/input/nih-chest-xrays/data/train_val_list.txt" \
  --nih_test_list "/kaggle/input/nih-chest-xrays/data/test_list.txt" \
  --output_dir "/kaggle/working/unimiss_nih" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

Outputs:

- `last.pth`: final epoch weights.
- `best.pth`: best validation/test score weights.
- `checkpoint.pth`: full training checkpoint for resuming, including model, optimizer, epoch and best score.
- `best_metrics.json`: metrics for the epoch that produced `best.pth`.
- `metrics.jsonl`: one JSON metrics row per epoch.
- `training_curves.png`: training loss, learning-rate, and validation/test metric curves.

On Kaggle, `--num_workers 4` is usually a good starting point for the 4 CPU-thread runtime. If the notebook hangs or DataLoader prints worker warnings, reduce it to `--num_workers 2`.

NIH is a multi-label task, so single-label `accuracy` is not used. The script reports `exact_match_accuracy`, where all 15 labels must match for an image, and `label_accuracy`, which averages correctness across all image-label pairs.

Evaluation artifacts are saved by default. Add `--no_plots` to disable them.

For `covid` and `covid_qu_ex`, the script writes:

- `latest_eval_confusion_matrix.png` and `latest_eval_confusion_matrix.csv`
- `latest_eval_roc_curves.png`
- `latest_eval_pr_curves.png`
- `latest_eval_classification_report.txt`
- `latest_eval_metrics.json`
- `latest_eval_predictions.npz`

For `nih`, the script writes:

- `latest_eval_per_class_auc.png`
- `latest_eval_per_class_ap.png`
- `latest_eval_label_frequency.png`
- `latest_eval_micro_roc.png`
- `latest_eval_micro_pr.png`
- `latest_eval_classification_report.txt`
- `latest_eval_metrics.json`
- `latest_eval_predictions.npz`

When running with `--eval_only`, artifact names use the `eval_` prefix instead of `latest_eval_`.

Training logs are printed as one compact line per epoch, for example:

```text
epoch=28 | train_loss=0.277916 | lr=7.32e-06 | accuracy=0.925751 | macro_auc=0.990672 | macro_ap=0.981508
```

Learning rate is scheduled by default with polynomial decay, matching the original downstream code style:

```bash
--lr_schedule poly --lr_power 0.9
```

Other options:

```bash
--lr_schedule constant
--lr_schedule cosine --min_lr 0.000001
```

## Multi-GPU Downstream Training

The flexible downstream script supports simple multi-GPU training with `torch.nn.DataParallel`. Use a comma-separated GPU list and add `--multi_gpu`:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task nih \
  --nih_root "/kaggle/input/nih-chest-xrays/data" \
  --nih_csv "/kaggle/input/nih-chest-xrays/data/Data_Entry_2017.csv" \
  --nih_train_list "/kaggle/input/nih-chest-xrays/data/train_val_list.txt" \
  --nih_test_list "/kaggle/input/nih-chest-xrays/data/test_list.txt" \
  --output_dir "/kaggle/working/unimiss_nih" \
  --pre_train \
  --pre_train_path "/kaggle/input/unimissplus/UniMissPlus.pth" \
  --gpu "0,1" \
  --multi_gpu \
  --batch_size 64 \
  --test_batch_size 64 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

`last.pth`, `best.pth`, and `checkpoint.pth` are saved without the `module.` prefix, so they can be loaded later on either one GPU or multiple GPUs.

## Resume Training

Use `--resume_path` to continue a previous training run from the saved full checkpoint:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/train_covid19" \
  --resume_path "/kaggle/working/train_covid19/checkpoint.pth" \
  --batch_size 32 \
  --test_batch_size 32 \
  --num_workers 4 \
  --epochs 30 \
  --learning_rate 0.0001
```

`--epochs` is the final target epoch count, not the number of extra epochs. For example, if `checkpoint.pth` was saved after epoch 9 and you pass `--epochs 30`, training resumes at epoch 10 and stops after epoch 29.

Use `--checkpoint_path` only for loading plain model weights such as `best.pth` or `last.pth` for evaluation/fine-tuning. Use `--resume_path` when you want to restore optimizer state, epoch number, best score and the learning-rate schedule position.

## Cross-Dataset Evaluation

Cross-dataset evaluation is supported when the source and target tasks use the same number of output classes.

The cleanest COVID cross-dataset setup is:

```text
COVID-19 Radiography:
  COVID
  Other = Lung_Opacity + Viral Pneumonia
  Normal

COVID-QU-Ex:
  COVID-19
  Non-COVID
  Normal
```

These two 3-class label spaces are aligned as:

```text
COVID      -> COVID-19
Other      -> Non-COVID
Normal     -> Normal
```

### Train on COVID-19 Radiography, Evaluate on COVID-QU-Ex

Train:

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

Evaluate on COVID-QU-Ex:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid_qu_ex \
  --covid_qu_ex_root "/kaggle/input/covid-qu-ex/COVID-QU-Ex" \
  --output_dir "/kaggle/working/eval_covid19_to_quex" \
  --checkpoint_path "/kaggle/working/train_covid19/best.pth" \
  --eval_only
```

### Train on COVID-QU-Ex, Evaluate on COVID-19 Radiography

Train:

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

Evaluate on COVID-19 Radiography:

```bash
python Downstream/2D/Cls/main_flexible.py \
  --task covid \
  --covid_mode other_normal \
  --covid_root "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
  --output_dir "/kaggle/working/eval_quex_to_covid19" \
  --checkpoint_path "/kaggle/working/train_quex/best.pth" \
  --eval_only
```

Avoid evaluating a model trained with the default COVID labels `COVID,Lung_Opacity,Normal` directly on COVID-QU-Ex, because `Lung_Opacity` and `Non-COVID` are not the same label definition.

