#!/bin/bash -l
#SBATCH --job-name sfb_convert
#SBATCH -o slurm_out-%j.output
#SBATCH -e slurm_out-%j.output
#SBATCH --partition med
#SBATCH --verbose

#SBATCH --mem-per-cpu 4G
#SBATCH --time 0-24:00:00
#SBATCH -n 1

conda activate general
python ptm_convert.py runs/merged_022_20*

