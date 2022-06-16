#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/brats_rerun/%x_%j_%N.out
#SBATCH --job-name s7-dfp-mrnet
#SBATCH --partition=debug
#SBATCH --dependency=singleton

newgrp cs-labname

module load LANG/PYTHON/3.7.6

cd /project/labname-lab/authorid/BRATS_IDH/code
git pull

virtualenv --no-download /localscratch/authorid/venv
source /localscratch/authorid/venv/bin/activate
pip install --no-index --upgrade pip
pip install -r /project/labname-lab/authorid/BRATS_IDH/code/requirement.txt


SEED=7

cd /project/labname-lab/authorid/BRATS_IDH/code
python xai_mrnet.py --machine solar -c /project/labname-lab/authorid/shortcut/BRATS_IDH/code/sh/mrnet_gethm/mrnet_xai_solar.json --model_path /project/labname-lab/authorid/ --seed $SEED --job fp |& tee -a /project/labname-lab/authorid/brats_rerun/s7_fp_mrnet.txt