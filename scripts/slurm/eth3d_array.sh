#!/bin/bash
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
$RUN python scripts/benchmark.py -d eth3d -c paper/$conf -m all -s $scene --save_sparse -t
$RUN python scripts/benchmark_gs.py -d eth3d -c $conf -m all -s $scene -t
echo "[$(date)] task $SLURM_ARRAY_TASK_ID done: $scene $conf"
