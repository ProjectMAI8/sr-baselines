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
├── metrics/               # Evaluation logs (PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA, FID)
├── run_diffbir.slurm      # SLURM job script for DiffBIR v2
├── run_diffbir_v1.slurm   # SLURM job script for DiffBIR v1
└── README.md
```

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

```bash
pip install -U huggingface_hub
huggingface-cli download Iceclear/StableSR-TestSets --repo-type dataset --local-dir data/StableSR-TestSets
```

Each test set contains `test_LR/` (input) and `test_HR/` (ground truth) folders.

## Pretrained weights

Weights are **not included**. Download them as described in the official repositories:

- **OSEDiff:** Stable Diffusion 2.1 base, OSEDiff weights (`osediff.pkl`), DAPE and RAM weights – see the [OSEDiff README](https://github.com/cswry/OSEDiff)
- **DiffBIR:** see the [DiffBIR README](https://github.com/XPixelGroup/DiffBIR) (weights go into `DiffBIR/weights/`)

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
python test_osediff.py \
  -i ../data/StableSR-TestSets/RealSRVal_crop128/test_LR \
  -o ../outputs/osediff/RealSR \
  --osediff_path preset/models/osediff.pkl \
  --pretrained_model_name_or_path PATH_TO_SD21_BASE \
  --ram_ft_path PATH_TO_DAPE \
  --ram_path PATH_TO_RAM
```

### DiffBIR (on the cluster via SLURM)

```bash
sbatch run_diffbir.slurm      # DiffBIR v2
sbatch run_diffbir_v1.slurm   # DiffBIR v1
```

## Results

All experiments at ×4 upscaling on the StableSR test sets (RealSRVal_crop128 and DrealSRVal_crop128).
↑ = higher is better, ↓ = lower is better. **Bold** = best per column.

Settings: OSEDiff uses 1 diffusion step; DiffBIR v2 uses 50 steps (spaced sampler, CFG scale 4).

### RealSR (RealSRVal_crop128)

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | CLIPIQA ↑ | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ | FID ↓ |
|---|---|---|---|---|---|---|---|---|---|
| OSEDiff | **25.15** | **0.7341** | **0.2920** | **0.2127** | 0.6690 | 5.65 | 69.09 | 0.6331 | **123.46** |
| DiffBIR v1 | 24.81 | 0.6595 | 0.3608 | 0.2311 | 0.6447 | **5.53** | 64.94 | 0.6232 | 128.09 |
| DiffBIR v2 | 24.83 | 0.6501 | 0.3650 | 0.2399 | **0.7052** | 5.84 | **69.28** | **0.6502** | 130.75 |

### DRealSR (DrealSRVal_crop128)

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | CLIPIQA ↑ | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ | FID ↓ |
|---|---|---|---|---|---|---|---|---|---|
| OSEDiff | **27.92** | **0.7836** | **0.2967** | **0.2163** | 0.6953 | 6.45 | 64.68 | 0.5896 | **135.56** |
| DiffBIR v1 | 26.82 | 0.6633 | 0.4497 | 0.2730 | 0.6370 | **6.27** | 60.86 | 0.5911 | 167.60 |
| DiffBIR v2 | 25.90 | 0.6245 | 0.4669 | 0.2882 | **0.7070** | 6.33 | **66.13** | **0.6209** | 180.37 |

### Summary

- **OSEDiff** is best on all fidelity and reference-based perceptual metrics (PSNR, SSIM, LPIPS, DISTS, FID) on both test sets, while needing only a single diffusion step.
- **DiffBIR v2** scores highest on most no-reference quality metrics (CLIPIQA, MUSIQ, MANIQA), i.e. its outputs look sharper and more detailed, but they are less faithful to the ground truth.
- **DiffBIR v1** achieves the best NIQE but is otherwise between the two.

### Log files

Full per-image and average metrics are in the `metrics/` folder:

| Log file | Model | Test set |
|---|---|---|
| `osediff_RealSR.log` | OSEDiff | RealSRVal_crop128 |
| `osediff_DRealSR.log` | OSEDiff | DrealSRVal_crop128 |
| `diffbir_v1_RealSRVal_crop128.log` | DiffBIR v1 | RealSRVal_crop128 |
| `diffbir_v1_DrealSRVal_crop128.log` | DiffBIR v1 | DrealSRVal_crop128 |
| `diffbir_v2_RealSRVal_crop128.log` | DiffBIR v2 | RealSRVal_crop128 |
| `diffbir_v2_DrealSRVal_crop128.log` | DiffBIR v2 | DrealSRVal_crop128 |

## Credits

This repository builds on the official implementations of
[OSEDiff](https://github.com/cswry/OSEDiff) and [DiffBIR](https://github.com/XPixelGroup/DiffBIR),
and uses the test sets from [StableSR](https://github.com/IceClear/StableSR).
All credit for the methods and datasets goes to the original authors; please cite their papers.
