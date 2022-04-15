#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-karray
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100  # Request GPU "generic resources" [--gres=gpu:]
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham. [--ntasks-per-node=32]
#SBATCH --mem=32G        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham. [--mem=127G ]
#SBATCH --time=00-12:58      # time (DD-HH:MM)
#SBATCH --output=slurm_outputs/dcgan-c-%u-%x-%j.txt
#SBATCH --mail-user=xz2yu@uwaterloo.ca
#SBATCH --mail-type=ALL
rotate=$VAR1
translate=$VAR2
scale=$VAR3
horizontal_flip=$VAR4
dataset=$VAR5
project_dir=~/projects/def-karray/xz2yu/thesis/data/
version=dataug-c-$rotate-$translate-$scale-$horizontal_flip-$dataset-$(date +%s%3N_%d_%m_%Y_%H_%M)
version_path=$SLURM_TMPDIR/$version
mkdir -p $version_path
echo "[STATUS] Created version: $version"
free -g
nvidia-smi
module load cuda
source ~/torch/bin/activate
echo "[STATUS] Python environment ready"
cd ~/projects/def-karray/xz2yu/thesis/code/new-data-aug
echo "[STATUS] Starting script at `date`"
python -u dcgan-c.py --dataset_dir $SLURM_TMPDIR --output_dir $version_path --transforms $rotate $translate $scale $horizontal_flip --dataset $dataset
echo "[STATUS] Script completed at `date`" 
tar -cjf ${project_dir}${version}.tar -C $version_path $(ls $version_path)
echo "[STATUS] Copied over outputs"
deactivate
echo "[STATUS] Deactivate python environment. EXITING ..."
