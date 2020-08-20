#!/bin/bash -l
#SBATCH --job-name merge22_setup
#SBATCH -o slurm_out-%j.output
#SBATCH -e slurm_out-%j.output
#SBATCH --partition med
#SBATCH --verbose

# Might have been just the python process.
# 8G makes scheduling slow even when just for one node
# but 4G crashed with oom
#SBATCH --mem-per-cpu 6G
#SBATCH --time 01:00:00
#SBATCH -n 1

conda activate general
# full_run will invoke merged_sun, which gets the run configured and
# then uses srun to call sun.

python full_run.py

