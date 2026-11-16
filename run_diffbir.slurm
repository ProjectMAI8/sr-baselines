#!/bin/bash
#SBATCH --job-name=diffbir_eval
#SBATCH --partition=L40s
#SBATCH --account=rohrbach
#SBATCH --qos=l40s
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
WORK=$HOME/sr_project
DATA=$WORK/data/StableSR-TestSets/StableSR_testsets
OUT=$WORK/outputs/diffbir_v2

module load miniconda3
eval "$(conda shell.bash hook)"
export PYTHONNOUSERSITE=1
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi

# ---- inference: DiffBIR v2, 50 steps ----
conda activate DiffBIR
cd $WORK/DiffBIR
for SET in RealSRVal_crop128 DrealSRVal_crop128; do
  python -u inference.py --task sr --upscale 4 --version v2 --sampler spaced --steps 50 \
    --captioner none --pos_prompt '' \
    --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
    --cfg_scale 4 --input $DATA/$SET/test_LR --output $OUT/$SET \
    --device cuda --precision fp32
done

# ---- metrics: same OSEDiff script as before ----
conda activate OSEDiff
cd $WORK/OSEDiff
for SET in RealSRVal_crop128 DrealSRVal_crop128; do
  echo "===== $SET ====="
  python test_metrics.py --inp_imgs $OUT/$SET --gt_imgs $DATA/$SET/test_HR --log $OUT/${SET}_metrics
done
