#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=6-24:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/localscratch/authorid/log/train_model/s10_%A.out
#SBATCH --job-name s10
#SBATCH --partition=long

newgrp cs-labname

module load LANG/PYTHON/3.7.6

if [ ! -f "/localscratch/authorid/" ]
then
  mkdir /localscratch/authorid/
else
  echo "/localscratch/authorid/ exists"
fi

cd /localscratch/authorid/

if [ ! -f "/localscratch/authorid/results_brats_rerun/trained_models" ]
then
  cp /project/labname-lab/authorid/results_brats_rerun.zip
  unzip results_brats_rerun.zip
fi

if [ ! -f "/localscratch/authorid/dld_data/brats2020" ]
then
  mkdir /localscratch/authorid/dld_data
  cp -r /project/labname-lab/authorid/dld_data/brats2020 /localscratch/authorid/dld_data
fi

if [ ! -f "/localscratch/authorid/BRATS_IDH" ]
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

mkdir /localscratch/authorid/log
SEED=10

cd /localscratch/authorid/BRATS_IDH/code
python xai_pipeline.py --config sh/xai_solar_plain2_BRATS_HGG_bratsrerun.json --fold 1 --seed $SEED --bs 1 --ts cc --job gethm
cp -r /localscratch/authorid/log