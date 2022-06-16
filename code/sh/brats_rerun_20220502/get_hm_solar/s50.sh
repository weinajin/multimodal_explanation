#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=6-24:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/brats_rerun/BRATS_HGG/s50_hm_fold_1/%x_%N_%j.out
#SBATCH --error=/project/labname-lab/authorid/brats_rerun/BRATS_HGG/s50_hm_fold_1/%x_%N_%j.err
#SBATCH --job-name s50-gethm
#SBATCH --partition=long

newgrp cs-labname

module load LANG/PYTHON/3.7.6

cd /project/labname-lab/authorid/BRATS_IDH/code
git pull

virtualenv --no-download /localscratch/authorid/venv
source /localscratch/authorid/venv/bin/activate
pip install --no-index --upgrade pip
pip install -r /project/labname-lab/authorid/BRATS_IDH/code/requirement.txt

SEED=50

cd /project/labname-lab/authorid/BRATS_IDH/code
python xai_pipeline.py --config sh/xai_solar_plain2_BRATS_HGG_bratsrerun.json --fold 1 --seed $SEED --bs 1 --ts solar --job gethm -r /project/labname-lab/authorid/brats_rerun/BRATS_HGG/s50_hm_fold_1/ |& tee -a /project/labname-lab/authorid/brats_rerun/BRATS_HGG/s50_hm_fold_1/s50_gethm.txt
