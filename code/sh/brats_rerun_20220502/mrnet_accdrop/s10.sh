#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-20:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/localscratch/authorid/%x_%j_%N.out
#SBATCH --error=/localscratch/authorid/%x_%j_%N.err
#SBATCH --job-name s10_accdrop_bgnb_mrnet
#SBATCH --partition=long

newgrp cs-labname

module load LANG/PYTHON/3.7.6

cd /project/labname-lab/authorid/BRATS_IDH/code

virtualenv --no-download /localscratch/authorid/venv
source /localscratch/authorid/venv/bin/activate
pip install --no-index --upgrade pip
pip install -r /project/labname-lab/authorid/BRATS_IDH/code/requirement.txt

SEED=10

cd /project/labname-lab/authorid/BRATS_IDH/code
python xai_mrnet.py --machine solar -c /project/labname-lab/authorid/shortcut/BRATS_IDH/code/sh/mrnet_gethm/mrnet_xai_solar.json --model_path /project/labname-lab/authorid/ --seed $SEED --job acc_drop_bg |& tee -a /localscratch/authorid/s10_accdrop_bgnb_mrnet.txt
python xai_mrnet.py --machine solar -c /project/labname-lab/authorid/shortcut/BRATS_IDH/code/sh/mrnet_gethm/mrnet_xai_solar.json --model_path /project/labname-lab/authorid/ --seed $SEED --job acc_drop_nb |& tee -a /localscratch/authorid/s10_accdrop_bgnb_mrnet.txt

cp /localscratch/authorid/s10_accdrop_bgnb_mrnet*.* /project/labname-lab/authorid/brats_rerun/