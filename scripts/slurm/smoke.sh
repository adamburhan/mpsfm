#!/bin/bash
# Cluster smoke test for the mpsfm sif. Run inside salloc on a GPU node:
#   bash scripts/slurm/smoke.sh $SCRATCH/mpsfm/mpsfm_ab45bc79.sif
set -e
SIF=${1:?usage: smoke.sh <path-to-sif>}

RUNNER=$(command -v apptainer || true)
if [ -z "$RUNNER" ]; then
    module load singularity 2>/dev/null || true
    RUNNER=$(command -v singularity || true)
fi
[ -z "$RUNNER" ] && { echo "no apptainer/singularity found"; exit 1; }
echo "using: $RUNNER ($($RUNNER --version))"

exec "$RUNNER" exec --nv --cleanenv --no-home \
    -B "$HOME/repos/mpsfm:/mpsfm" \
    -B /network/scratch/a/adam.burhan \
    --pwd /mpsfm \
    "$SIF" bash /mpsfm/scripts/slurm/smoke_inner.sh
