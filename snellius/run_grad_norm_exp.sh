#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --job-name=gradNorm
#SBATCH --output=slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd $HOME/neuralsymbolic

echo "Running gradient norm experiment script with env‐specific Python..."
# direct call to env's Python
/home/dlindberg/.conda/envs/eiai2025/bin/python src/grad_norm_exp.py

echo "Experiment completed."