#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/shortcut/log/MRNet/seed/s8_%A.out
#SBATCH --job-name s8_MRNet
#SBATCH --partition=long

module load LANG/PYTHON/3.7.6
source /home/authorid/brain/bin/activate

LR='0.000007'
SEED=8
mkdir /project/labname-lab/authorid/shortcut/log/MRNet/seed/$SEED

cd /project/labname-lab/authorid/shortcut/BRATS_IDH/MRNet
python train.py --rundir /project/labname-lab/authorid/shortcut/log/MRNet/seed/$SEED --task meniscus  --seed $SEED --learning_rate $LR