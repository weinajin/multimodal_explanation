#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/authorid/shortcut/log/MRNet/lr/000009_%A.out
#SBATCH --job-name 000009_lr
#SBATCH --account=rrg-PIID

module load python/3.7
module load openslide

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip

pip install -r /scratch/authorid/BRATS_IDH/code/requirement.txt

LR='0.000009'
mkdir /scratch/authorid/shortcut/log/MRNet/lr/$LR

cd /scratch/authorid/shortcut/MRNet/
python train.py --rundir /scratch/authorid/shortcut/log/MRNet/lr/$LR --task meniscus  --seed 6 --learning_rate $LR