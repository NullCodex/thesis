#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-karray
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100  # Request GPU "generic resources" [--gres=gpu:]
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham. [--ntasks-per-node=32]
#SBATCH --mem=32G        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham. [--mem=127G ]
#SBATCH --time=1-23:58      # time (DD-HH:MM)
#SBATCH --output=slurm_outputs/subset-dcgan-accuracy-%u-%x-%j.txt
#SBATCH --mail-user=xz2yu@uwaterloo.ca
#SBATCH --mail-type=ALL
project_dir=~/projects/def-karray/xz2yu/thesis/data/
version=subset-data-aug-accuracy-$(date +%d_%m_%Y_%H_%M)
version_path=$SLURM_TMPDIR/$version
generated=$VAR1
generated_path=$SLURM_TMPDIR/$generated
dataset=$VAR2
comparison=$VAR3
mkdir -p $version_path
echo "[STATUS] Created version: $version"
free -g
nvidia-smi
module load cuda
unzip -q ${project_dir}${generated}.zip -d $SLURM_TMPDIR/
echo "[STATUS] Created data directory"
source ~/torch/bin/activate
echo "[STATUS] Python environment ready"
cd ~/projects/def-karray/xz2yu/thesis/code/new-data-aug
echo "[STATUS] Starting script at `date`"
python -u new-accuracy.py --dataset_dir $SLURM_TMPDIR --output_dir $version_path --generated_dir $generated_path --dataset $dataset --comparison $comparison
echo "[STATUS] Script completed at `date`" 
tar -cjf ${project_dir}${version}.tar -C $version_path $(ls $version_path)
echo "[STATUS] Copied over outputs"
deactivate
echo "[STATUS] Deactivate python environment. EXITING ..."
