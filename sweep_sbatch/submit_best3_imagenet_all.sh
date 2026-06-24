#!/bin/bash
# Submit best 3 imagenet runs for the given radii.
# Usage: bash sweep_sbatch/submit_best3_imagenet_all.sh 0 0.1 0.5 1 3 5
# If no arguments are given, all radii are submitted.

VALID_RADII=(0 0.1 0.5 1 3 5)

if [ "$#" -eq 0 ]; then
    RADII=("${VALID_RADII[@]}")
else
    RADII=("$@")
fi

for radius in "${RADII[@]}"; do
    found=false
    for v in "${VALID_RADII[@]}"; do
        if [ "$radius" = "$v" ]; then
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "WARNING: radius $radius has no sbatch files, skipping."
        continue
    fi
    echo "Submitting radius $radius ..."
    sbatch sweep_sbatch/best3_imagenet_radius_${radius}_run1.sbatch
    sbatch sweep_sbatch/best3_imagenet_radius_${radius}_run2.sbatch
    sbatch sweep_sbatch/best3_imagenet_radius_${radius}_run3.sbatch
done
