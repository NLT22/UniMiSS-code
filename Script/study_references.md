# Study Reference Registry

Every method included in the Vietnamese X-ray evaluation should have a citation
entry here. Use the `citation_key` values in result tables and report drafts.

| Citation Key | Citation / Title | Method Role | URL / DOI | Used In Report |
|---|---|---|---|---|
| Xie2024_UniMiSSPlus | Xie et al., *UniMiSS+: Universal Medical Self-Supervised Learning From Cross-Dimensional Unpaired Data*, TPAMI 2024 | Main project method; CT/DRR extension basis | https://pubmed.ncbi.nlm.nih.gov/39083391/ | Main method; future CT-to-DRR extension |
| Xie2022_UniMiSS | Xie et al., *Universal Medical Self-Supervised Learning via Breaking Dimensionality Barrier*, ECCV 2022 | Background for UniMiSS/UniMiSS+ cross-dimensional pretraining | https://arxiv.org/abs/2112.09356 | Related work; method background |
| Cohen2022_TorchXRayVision | Cohen et al., *TorchXRayVision: A library of chest X-ray datasets and models*, MIDL/PMLR 2022 | CXR-specific pretrained DenseNet121 baseline | https://proceedings.mlr.press/v172/cohen22a.html | Baseline comparison |
| Sellergren2022_CXRTransfer | Sellergren et al., *Simplified Transfer Learning for Chest Radiography Models Using Less Data*, Radiology 2022 | Justifies pretrained CXR models under limited-data settings | https://pubs.rsna.org/doi/abs/10.1148/radiol.212482 | Data limitation and transfer-learning rationale |
| PerezGarcia2025_RAD_DINO | Perez-Garcia et al., *Exploring scalable medical image encoders beyond text supervision*, Nature Machine Intelligence 2025 | Modern open-weight CXR/biomedical vision encoder baseline | https://huggingface.co/microsoft/rad-dino | Recommended modern frozen-embedding comparison |
| Codella2024_MedImageInsight | Codella et al., *MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging*, arXiv 2024 | Strong modern medical imaging embedding baseline; reported strong CXR performance | https://arxiv.org/abs/2410.06542 | Optional if setup/access is practical |
| Google_CXRFoundation_ModelCard | Google Health AI Developer Foundations, *CXR Foundation Model Card* | Strong CXR-specific embedding model; ELIXR v2.0 optional baseline | https://developers.google.com/health-ai-developer-foundations/cxr-foundation/model-card | Optional if setup/access is practical |
| Li2025_FoundationEmbeddings | Li et al., *From Embeddings to Accuracy: Comparing Foundation Models for Radiographic Classification*, arXiv 2025 | Justifies frozen embeddings plus lightweight classifiers for comparison | https://arxiv.org/abs/2505.10823 | Evaluation design |
| Shin2025_CXRFoundationBenchmark | Shin et al., *Benchmarking CXR Foundation Models With Publicly Available MIMIC-CXR and NIH-CXR14 Datasets*, arXiv 2025 | Foundation-model comparison context for CXR classification | https://arxiv.org/abs/2512.06014 | Related work; optional foundation-model comparison |
| Google_MedSigLIP_ModelCard | Google Health AI Developer Foundations, *MedSigLIP Model Card* | Optional broad medical image-text embedding baseline | https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card | Optional baseline |
| Russakovsky2015_ImageNet | Russakovsky et al., *ImageNet Large Scale Visual Recognition Challenge*, IJCV 2015 | Generic non-medical pretraining source | https://www.image-net.org/challenges/LSVRC/ | ImageNet baseline |
| Huang2017_DenseNet | Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017 | DenseNet121 architecture citation | https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html | DenseNet baselines |
| He2016_ResNet | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 | ResNet50 architecture citation | https://arxiv.org/abs/1512.03385 | ResNet baseline |
| TRIPOD_AI | TRIPOD+AI reporting guideline | Transparent reporting of prediction-model study design | https://www.tripod-statement.org/ | Reporting guideline |

## Current Comparison Methods

| Method Name In Code | Citation Key | Pretraining Source | Adaptation |
|---|---|---|---|
| `rad_dino` | PerezGarcia2025_RAD_DINO | Self-supervised CXR / biomedical image pretraining | Frozen RAD-DINO CLS embeddings + logistic regression |
| `torchxrayvision_densenet121` | Cohen2022_TorchXRayVision | Chest X-ray datasets through TorchXRayVision | Frozen DenseNet121 embeddings + logistic regression |
| `imagenet_densenet121` | Russakovsky2015_ImageNet; Huang2017_DenseNet | ImageNet ILSVRC | Frozen DenseNet121 embeddings + logistic regression |
| `imagenet_resnet50` | Russakovsky2015_ImageNet; He2016_ResNet | ImageNet ILSVRC | Frozen ResNet50 embeddings + logistic regression |
| `medsiglip` | Google_MedSigLIP_ModelCard | MedSigLIP medical image-text pretraining | Frozen image embeddings + logistic regression |
| `unimissplus_finetune` | Xie2024_UniMiSSPlus; Xie2022_UniMiSS | UniMiSS+ / UniMiSS pretrained weights | Fine-tuning in UniMiSSPlus downstream code |

## Method Status For This Study

| Method | Status | Reason |
|---|---|---|
| `rad_dino` | Recommended modern baseline | Open Hugging Face model; recent CXR-specialized encoder; used as the main frozen-feature sanity check against older DenseNet baselines. |
| `unimissplus_finetune` | Main project method | Matches the chosen UniMiSSPlus paper direction and supports later CT/DRR extension. |
| `torchxrayvision_densenet121` | Keep as CXR-specific conventional baseline | Older than current foundation models, but still a standard open CXR baseline/feature extractor. Do not present it as SOTA. |
| `imagenet_densenet121` / `imagenet_resnet50` | Keep as generic lower-bound baselines | Architectures are old, but ImageNet pretraining is a necessary non-medical reference point. Do not compare final claims only against ImageNet. |
| `medsiglip` | Optional broad medical baseline | Useful if setup is quick; not CXR-specific enough to replace RAD-DINO or CXR Foundation. |
| MedImageInsight / CXR Foundation | Optional stronger modern baselines | Recent benchmarks report strong embedding performance, but setup/access may take longer than the current project week. Add only if install and licensing are practical. |

## Rules For Adding A Method

- Do not add a method to final result tables unless it has a citation key here.
- Do not use random initialization as a comparison method for this study.
- Public datasets may be used as pretraining sources through public weights, but
  public images are not added to Vietnamese training folds.
- For binary evaluation, use Abnormal as the positive class.
