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

```
sr-baselines/
├── OSEDiff/               # OSEDiff code
├── DiffBIR/               # DiffBIR code
├── metrics/               # Evaluation results (PSNR, SSIM, LPIPS, ...)
├── run_diffbir.slurm      # SLURM job script for DiffBIR v2
├── run_diffbir_v1.slurm   # SLURM job script for DiffBIR v1
└── README.md
```

## Datasets

Datasets are **not included** in this repository due to size and licensing.
We used the real-world SR test sets released with StableSR:

| Dataset | Description | Source |
|---|---|---|
| RealSR | Real LR/HR pairs (Canon, Nikon), ×4 | [link] |
| DRealSR | Real LR/HR pairs (multiple DSLRs), ×4 | [link] |
| RealSRVal_crop128 | Cropped RealSR validation set | [link] |
| DrealSRVal_crop128 | Cropped DRealSR validation set | [link] |

Place the downloaded data in:

```
data/
├── RealSR/
├── DRealSR/
└── ...
```

## Pretrained weights

Weights are **not included**. Download them as described in the official repositories:
- OSEDiff: see the OSEDiff README (Stable Diffusion 2.1 base + OSEDiff + DAPE weights)
- DiffBIR: see the DiffBIR README (weights are downloaded into `DiffBIR/weights/`)

## Setup

```bash
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
```

## Running

### OSEDiff
```bash
cd OSEDiff
[your OSEDiff inference command]
```

### DiffBIR (on the cluster via SLURM)
```bash
sbatch run_diffbir.slurm      # DiffBIR v2
sbatch run_diffbir_v1.slurm   # DiffBIR v1
```

## Results

All results at ×4 upscaling. Full logs are in the `metrics/` folder.

| Model | Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| OSEDiff | RealSR | [ ] | [ ] | [ ] |
| DiffBIR v2 | RealSR | [ ] | [ ] | [ ] |
| DiffBIR v2 | DRealSR | [ ] | [ ] | [ ] |

## Credits

This repository builds on the official implementations of
[OSEDiff](https://github.com/cswry/OSEDiff) and [DiffBIR](https://github.com/XPixelGroup/DiffBIR).
All credit for the methods goes to the original authors; please cite their papers.
