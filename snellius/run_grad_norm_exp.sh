#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate eiai2025

cd $HOME/neuralsymbolic

echo "Running gradient norm experiment script..."

python src/grad_norm_exp.py

echo "Experiment completed. Check the output files for results."