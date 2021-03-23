#!/bin/bash -l
#SBATCH --job-name sfb_ptm22b
#SBATCH -o slurm_out-%j.output
#SBATCH -e slurm_out-%j.output
#SBATCH --partition high2
#SBATCH --verbose

#SBATCH --mem-per-cpu 2G
#SBATCH --time 5-00:00:00

# This way is going to be tough to schedule...
#SBATCH -n 32
#SBATCH -N 1

conda activate general
# then uses srun to call sun.
export OMP_NUM_THREADS=32

python batch_run7.py

