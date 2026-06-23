#!/bin/bash
# Submit best 3 runs for imagenet radius_0.5 and radius_1 in parallel
sbatch sweep_sbatch/best3_imagenet_radius_0.5_run1.sbatch
sbatch sweep_sbatch/best3_imagenet_radius_0.5_run2.sbatch
sbatch sweep_sbatch/best3_imagenet_radius_0.5_run3.sbatch
sbatch sweep_sbatch/best3_imagenet_radius_1_run1.sbatch
sbatch sweep_sbatch/best3_imagenet_radius_1_run2.sbatch
sbatch sweep_sbatch/best3_imagenet_radius_1_run3.sbatch
