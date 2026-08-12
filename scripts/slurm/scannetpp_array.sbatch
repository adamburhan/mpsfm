#!/bin/bash
# ScanNet++ main-table campaign: combined SfM + 3DGS per (scene, arm).
# 21 scenes (all val scenes with <=300 images, size rule fixed before results)
# x 2 arms = 42 tasks. Both stages are idempotent: requeue a task and it
# fast-forwards past completed work.
#
#   sbatch scripts/slurm/scannetpp_array.sbatch
#   sbatch --array=7 scripts/slurm/scannetpp_array.sbatch   # retry one task
#
#SBATCH --job-name=mpsfm-snpp
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-41%16
#SBATCH --output=/network/scratch/a/adam.burhan/logs/mpsfm_snpp/%A_%a.out

set -euo pipefail

SCENES=(
    0d2ee665be f3685d06a9 5eb31827b7 e398684d27 bcd2436daf
    40aec5fffa 3864514494 27dd4da69e a8bf42d646 45b0dac5e3
    b0a08200c9 21d970d8de 5ee7c22ba0 7bc286c1b6 31a2c91c43
    1ada7a0617 286b55a2bf c50d2d1d42 f3d64c30f8 bde1e479ad
    578511c8a9
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
