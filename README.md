# uncommon-retinal-disease-AI-OCT

A Swin Transformer–based framework for nine-class optical coherence tomography (OCT) image classification, systematically evaluating classification performance, generalization, predictive reliability, and explainability under real-world class-imbalanced conditions.

## Nine-Class Taxonomy

| Index | Class | Description |
|-------|-------|-------------|
| 0 | NORMAL | Normal retina |
| 1 | AMD | Age-related macular degeneration |
| 2 | DR | Diabetic retinopathy |
| 3 | ERM | Epiretinal membrane |
| 4 | MH | Macular hole |
| 5 | RRD | Rhegmatogenous retinal detachment |
| 6 | RVO | Retinal vein occlusion |
| 7 | RAO | Retinal artery occlusion |
| 8 | CSC | Central serous chorioretinopathy |

## Environment Setup

```bash
# Create conda environment (Python 3.10 recommended)
conda create -n ai-oct python=3.10 -y
conda activate ai-oct

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies**: PyTorch 2.0+, PyTorch Lightning 2.0+, MONAI, timm, SimpleITK, OpenCV, pytorch-grad-cam.

## Data Preparation

Training data is organized in an Excel spreadsheet with the following columns:

| Column | Description |
|--------|-------------|
| `process_img_path` | Absolute path to the preprocessed image (supports .mha/.mhd/.png/.jpg/.npz) |
| `label_dict` | Disease label name (must be one of the nine classes) |
| `datatag` | Data split tag: `train` / `valid` / `test` |
| `dataset` | Dataset source name (used for grouping during external testing) |
| `case_id` | Case ID (optional, for visualization annotations) |
| `black` | Quality filter flag (optional; images with black=1 are filtered out) |

Image preprocessing pipeline: orientation correction (`detect_orientation`) → adaptive resize (`adaptive_resize` to 256×448) → intensity normalization (`ScaleIntensity`, 0–1).

**Ruler noise template**: Training uses `AddRulerNoised` data augmentation. Place the ruler template image at `./assets/ruler_mask.png`. This file is a binary template (not included in the repository); users should prepare it separately or modify the path in `ds.py`.

## Training

```bash
# Standard Swin-T training
python train.py \
    --train_xlsx /path/to/train.xlsx \
    --valid_xlsx /path/to/valid.xlsx \
    --test_xlsx /path/to/test.xlsx \
    --output_dir ./outputs/swin_t \
    --model_name swin \
    --batch_size 32 \
    --max_epochs 100 \
    --lr 1e-4

# Baseline model training (optional)
python train.py --model_name resnet50 --output_dir ./outputs/resnet50
python train.py --model_name densenet121 --output_dir ./outputs/densenet121
python train.py --model_name convnext_tiny --output_dir ./outputs/convnext_tiny
python train.py --model_name vit_small --output_dir ./outputs/vit_small

# Ablation: remove shifted-window (SW-MSA) mechanism
python train.py \
    --train_xlsx /path/to/train.xlsx \
    --valid_xlsx /path/to/valid.xlsx \
    --test_xlsx /path/to/test.xlsx \
    --output_dir ./outputs/swin_noshift \
    --model_name swin \
    --no_shift
```

During training, the best checkpoint (`best.ckpt`) and the last checkpoint (`last.ckpt`) are saved automatically. After training completes, the model is evaluated on the test set.

## External Testing

```bash
# Full external test (with visualization and CAM heatmaps)
python extest.py \
    --xlsx /path/to/external_test.xlsx \
    --ckpt /path/to/best.ckpt \
    --output ./external_results \
    --batch_size 64 \
    --cam_method layercam

# Inference-only (skip visualization and heatmaps, faster)
python extest.py \
    --xlsx /path/to/external_test.xlsx \
    --ckpt /path/to/best.ckpt \
    --output ./external_results \
    --no_vis --no_cam
```

**External test outputs**:
- `Dataset_Summary.csv`: Macro-averaged metrics across datasets
- `<dataset_name>/ConfusionMatrix.png`: Rectangular confusion matrix (true-class rows × all 9 predicted columns)
- `<dataset_name>/PerClassBar.png`: Per-class Precision/Recall/F1 bar chart
- `<dataset_name>/Performance.csv`: Detailed per-class metrics
- `Overall/detailed_predictions.csv`: Per-sample prediction records
- `Overall/vis_images/`: Per-sample prediction visualization (GT/Pred/Top-3 annotations)
- `Overall/gradcam/`: LayerCAM explainability heatmaps (correct/wrong samples separated)

## Directory Structure

```
.
├── ds.py              # Dataset class: loading, preprocessing, augmentation, weighted sampling
├── model.py           # Model definitions: Swin-T + 4 baselines + noshift ablation monkey patch
├── train.py           # Training script: PyTorch Lightning, argparse-configurable
├── wrapper.py         # Lightning wrapper: train/val/test steps, metrics, visualization
├── extest.py          # External test script: multi-dataset evaluation, visualization, CAM
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Models

### Main Model: Swin Transformer-Tiny

- Input size: 256×448 (single-channel grayscale, replicated to 3 channels)
- Pretrained weights: ImageNet-1k (timm library)
- Classification head: Linear(768, 9)

### Baseline Models

| Model | timm Name | Params |
|-------|-----------|--------|
| ResNet-50 | resnet50 | ~25M |
| DenseNet-121 | densenet121 | ~8M |
| ConvNeXt-Tiny | convnext_tiny | ~28M |
| ViT-Small | vit_small_patch16_224 | ~22M |

### Ablation: Shifted-Window Mechanism

A monkey patch replaces `SwinTransformerBlock._attn` to skip the cyclic shift (`torch.roll`), retaining only window-based multi-head self-attention (W-MSA). This ablation assesses the contribution of shifted-window attention to classification performance. Enable with the `--no_shift` flag during training.

## Citation

If this code contributes to your research, please cite the associated paper.

## License

This project is intended for academic research purposes only.
