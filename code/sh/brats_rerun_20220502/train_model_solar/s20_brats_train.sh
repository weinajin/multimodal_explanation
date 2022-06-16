#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-24:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/brats_rerun/%x_%j_%N.out
#SBATCH --job-name s43-Train
#SBATCH --partition=long


newgrp cs-labname

module load LANG/PYTHON/3.7.6

cd /project/labname-lab/authorid/BRATS_IDH/code
git pull

virtualenv --no-download /localscratch/authorid/venv
source /localscratch/authorid/venv/bin/activate
pip install --no-index --upgrade pip
pip install -r /project/labname-lab/authorid/BRATS_IDH/code/requirement.txt


SEED=43

cd /project/labname-lab/authorid/BRATS_IDH/code
python train.py --config sh/config_solar_plain2_BRATS_HGG.json --fold 1 --seed $SEED |& tee -a /project/labname-lab/authorid/brats_rerun/s43_train.txt