#!/bin/bash
#SBATCH --job-name=15M_5
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --nodelist=sh02-04n[39-44]
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=48:00:00
#SBATCH -p owners
#SBATCH --mail-user=eantoniu@stanford.edu
#SBATCH --mail-type=ALL
srun hostname

module purge
module load math
module load chemistry
module load python/3.6.1

ulimit -s unlimited

python3 atom2vec_semi_supervise_v3_batched_2020_hyperparams.py

echo "Job finished at"
date
