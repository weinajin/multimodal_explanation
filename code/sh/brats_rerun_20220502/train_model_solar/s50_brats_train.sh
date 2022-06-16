#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=6-24:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/brats_rerun/%x_%N_%j.out
#SBATCH --error=/project/labname-lab/authorid/brats_rerun/%x_%N_%j.err
#SBATCH --job-name s50
#SBATCH --partition=long
#SBATCH --nodelist=cs-venus-01

newgrp cs-labname
module load LANG/PYTHON/3.7.6

if [! -d "/localscratch/authorid/dld_data/brats2020" ]
then
  mkdir /localscratch/authorid/dld_data
  cp -r /project/labname-lab/authorid/dld_data/brats2020 /localscratch/authorid/dld_data
fi

if [! -d "/localscratch/authorid/BRATS_IDH" ]
then
  cd /localscratch/authorid
  git clone git@github.com:authoridin/BRATS_IDH.git
else
  git pull
fi

virtualenv --no-download /localscratch/authorid/venv
source /localscratch/authorid/venv/bin/activate
pip install --no-index --upgrade pip
pip install -r /localscratch/authorid/BRATS_IDH/code/requirement.txt
source /localscratch/authorid/venv/bin/activate

SEED=50

python train.py --config sh/config_solar_plain2_BRATS_HGG.json --fold 1 --seed $SEED