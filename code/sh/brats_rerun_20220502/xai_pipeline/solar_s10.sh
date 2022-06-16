#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/localscratch/authorid/log/s10_hm_fold_1/%x_%N_%j.out
#SBATCH --error=/localscratch/authorid/log/s10_hm_fold_1/%x_%N_%j.err
#SBATCH --job-name s10-ppl
#SBATCH --partition=debug

newgrp cs-labname

module load LANG/PYTHON/3.7.6

if [ ! -f "/localscratch/authorid/" ]
then
  mkdir /localscratch/authorid/
else
  echo "/localscratch/authorid/ exists"
fi

if [ ! -f "/localscratch/authorid/log" ]
then
  mkdir /localscratch/authorid/log
else
  echo "/localscratch/authorid/log exists"
fi

cd /project/labname-lab/authorid/BRATS_IDH/code
git pull

cd /localscratch/authorid/
if [ -f "/localscratch/authorid/BRATS_IDH" ]
then
  rm -rf /localscratch/authorid/BRATS_IDH
fi

cp -r /project/labname-lab/authorid/BRATS_IDH/  /localscratch/authorid/

if [ ! -f "/localscratch/authorid/venv" ]
then
  virtualenv --no-download /localscratch/authorid/venv
  source /localscratch/authorid/venv/bin/activate
  pip install --no-index --upgrade pip
  pip install -r /localscratch/authorid/BRATS_IDH/code/requirement.txt
fi

SEED=10

cd /localscratch/authorid/BRATS_IDH/code
python xai_pipeline.py --config sh/xai_solar_plain2_BRATS_HGG.json --fold 1 --seed $SEED --bs 1 --ts solar --job pipeline -r /project/labname-lab/authorid/brats_rerun/BRATS_HGG/s10_hm_fold_1/ |& tee -a /project/labname-lab/authorid/brats_rerun/BRATS_HGG/s10_hm_fold_1/s10_pipeline.txt
