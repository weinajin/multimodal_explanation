#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/authorid/shortcut/log/MRNet/weight_decay/0_%A.out
#SBATCH --job-name wt0
#SBATCH --account=rrg-PIID

module load python/3.7
module load openslide

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip

pip install -r /scratch/authorid/BRATS_IDH/code/requirement.txt

LR='0.00001'
WT='0'

cd /scratch/authorid/shortcut/MRNet/
python train.py --rundir /scratch/authorid/shortcut/log/MRNet/weight_decay/$WT --task meniscus  --seed 6 --learning_rate $LR --weight_decay $WT