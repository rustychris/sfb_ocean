#!/bin/bash -l
#SBATCH --job-name sfb_merge22
#SBATCH -o slurm_out-%j.output
#SBATCH -e slurm_out-%j.output
#SBATCH --partition high
#SBATCH --verbose
# Not sure if I can select this -- does not appear to do anything
#SBATCH --requeue

# Got some messages that the process was killed due to out-of-memory
# Might have been just the python process.
# 4G makes scheduling slow
#SBATCH --mem-per-cpu 2G
#SBATCH --time 5-00:00:00
# don't request a single node.  with 16 cores and each node having
# 12 cores/24 threads, this will oversubscribe the core resources

# This way is going to be tough to schedule...
#SBATCH -N 4
#SBATCH --ntasks-per-node 4

# Old way: I was doing this, but the heterogeneous jobs seem buggy.
#REM   First component has enough resources on a single node to run the setup script:
#REM  SBATCH -n 4 -N 1
#REM  SBATCH packjob
#REM  SBATCH --ntasks-per-core 1
#REM   Now I'm putting enough cores in the second component to run the full
#REM   mpi job
#REM  SBATCH -n 16 -N 1-12



conda activate general
# full_run will invoke merged_sun, which gets the run configured and
# then uses srun to call sun.

export OMPI_MCA_mpi_show_mca_params=all
export OMPI_MCA_mpi_abort_print_stack=true

# Let's see how the heteorgeneous stuff is reported. seems that there
# might be an issue where NTASKS is not reflecting the global count
export

python full_run.py

