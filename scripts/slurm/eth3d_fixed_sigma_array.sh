#!/bin/bash
#SBATCH --job-name=mpsfm-eth3d-fs
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-359%50
#SBATCH --output=/network/scratch/a/adam.burhan/logs/mpsfm_eth3d_fs/%A_%a.out

# Fixed-sigma battery on ETH3D: unimodal vs maxmix vs maxmix+null, sigma=0.03,
# integration off, across view-count regimes (sparse -> dense).
# Task -> (scene, conf, mode): scene = id % 24, conf = (id/24) % 3, mode = id/72.
# Modes ordered smallest-first so early tasks finish fast.

set -euo pipefail

SCENES=(
    botanical_garden
    boulders
    bridge
    courtyard
    delivery_area
    door
    electro
    exhibition_hall
    facade
    kicker
    lecture_room
    living_room
    lounge
    meadow
    observatory
    office
    old_computer
    pipes
    playground
    relief
    relief_2
    statue
    terrace
    terrains
)
CONFS=(
    repr-sp-lg_m3dv2-uni-cauchy003-noint
    repr-sp-lg_m3dv2-maxmix-cauchy003-noint
    repr-sp-lg_m3dv2-maxmix-null003-noint
)
MODES=(minimal leq5 leq10 leq30 all)

n=${#SCENES[@]}
scene=${SCENES[$((SLURM_ARRAY_TASK_ID % n))]}
conf=${CONFS[$(((SLURM_ARRAY_TASK_ID / n) % ${#CONFS[@]}))]}
mode=${MODES[$((SLURM_ARRAY_TASK_ID / (n * ${#CONFS[@]})))]}

SCRATCH_ROOT=/network/scratch/a/adam.burhan
SIF=$SCRATCH_ROOT/mpsfm/mpsfm_ab45bc79.sif

# writable caches outside the read-only sif (see mpsfm-dev-setup notes)
export SINGULARITYENV_TORCH_HOME=$SCRATCH_ROOT/mpsfm/torch-cache
export SINGULARITYENV_CUPY_CACHE_DIR=$SCRATCH_ROOT/mpsfm/cupy_cache
export SINGULARITYENV_MPLCONFIGDIR=/tmp/mpl

module load singularity

RUN="singularity exec --nv --cleanenv --no-home \
    -B $HOME/repos/mpsfm:/mpsfm -B $SCRATCH_ROOT --pwd /mpsfm $SIF"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: scene=$scene conf=$conf mode=$mode"
$RUN python scripts/benchmark.py -d eth3d -c paper/$conf -m $mode -s $scene -t
echo "[$(date)] task $SLURM_ARRAY_TASK_ID done: $scene $conf $mode"
