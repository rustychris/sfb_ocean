#!/bin/bash -l
#SBATCH --job-name sfb_ptm22
#SBATCH -o slurm_out-%j.output
#SBATCH -e slurm_out-%j.output
#SBATCH --partition high2
#SBATCH --verbose

#SBATCH --mem-per-cpu 2G
#SBATCH --time 5-00:00:00

# This way is going to be tough to schedule...
#SBATCH -n 8
#SBATCH -N 1

conda activate general
# full_run will invoke merged_sun, which gets the run configured and
# then uses srun to call sun.
export OMP_NUM_THREADS=8

python batch_run5.py

