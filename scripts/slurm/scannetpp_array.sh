#!/bin/bash
# ScanNet++ combined SfM + 3DGS per (scene, arm). All 50 val scenes, ordered
# by image count: indices 0-20 small (<=300 imgs), 21-38 medium (300-900),
# 39-49 large (869-2355). Arm B = same order at +50.
#   sbatch --array=0-38,50-88%16 scripts/slurm/scannetpp_array.sh
#   sbatch --array=39-49,89-99%6 --time=96:00:00 --mem=128G scripts/slurm/scannetpp_array.sh
# Retry one task: sbatch --array=<idx> [overrides] scripts/slurm/scannetpp_array.sh
#
#SBATCH --job-name=mpsfm-snpp
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-99%16
#SBATCH --output=/network/scratch/a/adam.burhan/logs/mpsfm_snpp/%A_%a.out

set -euo pipefail

SCENES=(
    # small (<=300 images) — the pre-registered main-table subset
    0d2ee665be f3685d06a9 5eb31827b7 e398684d27 bcd2436daf
    40aec5fffa 3864514494 27dd4da69e a8bf42d646 45b0dac5e3
    b0a08200c9 21d970d8de 5ee7c22ba0 7bc286c1b6 31a2c91c43
    1ada7a0617 286b55a2bf c50d2d1d42 f3d64c30f8 bde1e479ad
    578511c8a9
    # medium (300-900)
    25f3b7a318 13c3e046d7 38d58a7a31 3e8bba0176 3f15a9266d
    5942004064 6115eddb86 7831862f02 7b6477cb95 825d228aec
    99fa5c25e1 a24f64f7fb a980334473 ac48a9b736 c4c04e6d6c
    c5439f4607 f2dc06b1d2 f9f95681fd
    # large (869-2355 images) — heavy SfM; submit with --time/--mem overrides
    d755b3d9d8 3db0a1c8f3 5748ce6f01 fb5a96b1a2 9071e139d9
    c49a8c6cff e7af285f7d cc5237fd77 acd95847c5 09c1414f1b
    5f99900f09
)
CONFS=(repr-sp-lg_m3dv2 repr-sp-lg_m3dv2-maxmix-comp)

n=${#SCENES[@]}
scene=${SCENES[$((SLURM_ARRAY_TASK_ID % n))]}
conf=${CONFS[$((SLURM_ARRAY_TASK_ID / n))]}

SCRATCH_ROOT=/network/scratch/a/adam.burhan
SIF=$SCRATCH_ROOT/mpsfm/mpsfm_ab45bc79.sif

# writable caches outside the read-only sif (see mpsfm-dev-setup notes)
export SINGULARITYENV_TORCH_HOME=$SCRATCH_ROOT/mpsfm/torch-cache
export SINGULARITYENV_CUPY_CACHE_DIR=$SCRATCH_ROOT/mpsfm/cupy_cache
export SINGULARITYENV_MPLCONFIGDIR=/tmp/mpl

module load singularity

RUN="singularity exec --nv --cleanenv --no-home \
    -B $HOME/repos/mpsfm:/mpsfm -B $SCRATCH_ROOT --pwd /mpsfm $SIF"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: scene=$scene conf=$conf"
$RUN python scripts/benchmark.py -d scannetpp -c paper/$conf -m all -s $scene --save_sparse -t
$RUN python scripts/benchmark_gs.py -d scannetpp -c $conf -m all -s $scene -t
echo "[$(date)] task $SLURM_ARRAY_TASK_ID done: $scene $conf"
