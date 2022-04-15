#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-karray
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100  # Request GPU "generic resources" [--gres=gpu:]
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham. [--ntasks-per-node=32]
#SBATCH --mem=32G        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham. [--mem=127G ]
#SBATCH --time=0-23:58      # time (DD-HH:MM)
#SBATCH --output=slurm_outputs/tgan-%u-%x-%j.txt
#SBATCH --mail-user=xz2yu@uwaterloo.ca
#SBATCH --mail-type=ALL
epochs=$VAR1
batch_size=$VAR2
seq_length=$VAR3
dataset=$VAR4
with_dtw=$VAR5
n_layers=$VAR6
project_dir=~/projects/def-karray/xz2yu/thesis/data/
version=tgan-$dataset-$with_dtw-$seq_length-$n_layers-$(date +%s%3N_%d_%m_%Y_%H_%M)
version_path=$SLURM_TMPDIR/$version
mkdir -p $version_path
echo "[STATUS] Created version: $version"
free -g
nvidia-smi
module load cuda
source ~/torch/bin/activate
echo "[STATUS] Python environment ready"
cd ~/projects/def-karray/xz2yu/thesis/code/time-series
mkdir -p $SLURM_TMPDIR/data
cp energy_data.csv $SLURM_TMPDIR/data
cp stock_data.csv $SLURM_TMPDIR/data
echo "[STATUS] Starting script at `date`"
python -u main.py --dataset_dir $SLURM_TMPDIR --output_dir $version_path --n_epochs $epochs --batch_size $batch_size --seq_length $seq_length --dataset $dataset --with_dtw $with_dtw --num_layers $n_layers
echo "[STATUS] Script completed at `date`" 
tar -cjf ${project_dir}${version}.tar -C $version_path $(ls $version_path)
echo "[STATUS] Copied over outputs"
deactivate
echo "[STATUS] Deactivate python environment. EXITING ..."
