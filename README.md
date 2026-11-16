# Super-Resolution Baselines – Project Lab Multimodal AI (TU Darmstadt)

Reproduction of diffusion-based image super-resolution models as baselines
for our Project Lab work on a new image super-resolution dataset.
All experiments were run on the ADA GPU cluster at TU Darmstadt.

## Models

| Model | Paper | Official code |
|---|---|---|
| OSEDiff | One-Step Effective Diffusion Network for Real-World Image Super-Resolution (NeurIPS 2024) | https://github.com/cswry/OSEDiff |
| DiffBIR (v1, v2) | DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior (ECCV 2024) | https://github.com/XPixelGroup/DiffBIR |

## Repository structure

sr-baselines/
├── OSEDiff/               # OSEDiff code
├── DiffBIR/               # DiffBIR code
├── metrics/               # Evaluation logs (PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA, FID)
├── run_diffbir.slurm      # SLURM job script for DiffBIR v2
├── run_diffbir_v1.slurm   # SLURM job script for DiffBIR v1
└── README.md

## Datasets

Datasets are **not included** in this repository due to size and licensing.
We used the real-world SR test sets released with StableSR, hosted on Hugging Face:
https://huggingface.co/datasets/Iceclear/StableSR-TestSets

| Dataset | Description |
|---|---|
| RealSRVal_crop128 | Center-cropped LR/HR pairs from RealSR v3 |
| DrealSRVal_crop128 | Center-cropped LR/HR pairs from DRealSR |

Original datasets: [RealSR](https://github.com/csjcai/RealSR) · [DRealSR](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution)
License: [S-Lab License 1.0](https://github.com/IceClear/StableSR/blob/main/LICENSE.txt)

Download:
bash
pip install -U huggingface_hub
huggingface-cli download Iceclear/StableSR-TestSets --repo-type dataset --local-dir data/StableSR-TestSets

Each test set contains `test_LR/` (input) and `test_HR/` (ground truth) folders.

## Pretrained weights

Weights are **not included**. Download them as described in the official repositories:
- **OSEDiff:** Stable Diffusion 2.1 base, OSEDiff weights (`osediff.pkl`), DAPE and RAM weights – see the [OSEDiff README](https://github.com/cswry/OSEDiff)
- **DiffBIR:** see the [DiffBIR README](https://github.com/XPixelGroup/DiffBIR) (weights go into `DiffBIR/weights/`)

## Setup

bash
# OSEDiff
cd OSEDiff
conda create -n osediff python=3.10 -y
conda activate osediff
pip install -r requirements.txt

# DiffBIR
cd ../DiffBIR
conda create -n diffbir python=3.10 -y
conda activate diffbir
pip install -r requirements.txt

## Running

### OSEDiff
bash
cd OSEDiff
python test_osediff.py \
  -i ../data/StableSR-TestSets/RealSRVal_crop128/test_LR \
  -o ../outputs/osediff/RealSR \
  --osediff_path preset/models/osediff.pkl \
  --pretrained_model_name_or_path <path to stable-diffusion-2-1-base> \
  --ram_ft_path <path to DAPE.pth> \
  --ram_path <path to ram_swin_large_14m.pth>


### DiffBIR (on the cluster via SLURM)
bash
sbatch run_diffbir.slurm      # DiffBIR v2
sbatch run_diffbir_v1.slurm   # DiffBIR v1

## Results

All experiments at ×4 upscaling. Full per-image and average metrics are in the `metrics/` folder:

| Log file | Model | Test set |
|---|---|---|
| `osediff_RealSR.log` | OSEDiff | RealSR |
| `diffbir_v2_RealSR.log` | DiffBIR v2 | RealSR |
| `diffbir_v2_DRealSR.log` | DiffBIR v2 | DRealSR |
| `diffbir_v2_RealSRVal_crop128.log` | DiffBIR v2 | RealSRVal_crop128 |
| `diffbir_v2_DrealSRVal_crop128.log` | DiffBIR v2 | DrealSRVal_crop128 |
| `diffbir_v1_RealSRVal_crop128.log` | DiffBIR v1 | RealSRVal_crop128 |
| `diffbir_v1_DrealSRVal_crop128.log` | DiffBIR v1 | DrealSRVal_crop128 |

Metrics: PSNR ↑, SSIM ↑, LPIPS ↓, DISTS ↓, CLIPIQA ↑, NIQE ↓, MUSIQ ↑, MANIQA ↑, FID ↓

## Credits

This repository builds on the official implementations of
[OSEDiff](https://github.com/cswry/OSEDiff) and [DiffBIR](https://github.com/XPixelGroup/DiffBIR),
and uses the test sets from [StableSR](https://github.com/IceClear/StableSR).
All credit for the methods and datasets goes to the original authors; please cite their papers.
