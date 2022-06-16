#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=6-24:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/localscratch/authorid/log/train_model/s30_%A.out
#SBATCH --job-name s30
#SBATCH --partition=long
#SBATCH --nodelist=cs-venus-04


module load LANG/PYTHON/3.7.6

cd /localscratch/authorid/BRATS_IDH/code
git pull

source /localscratch/authorid/venv/bin/activate


SEED=30

python train.py --config sh/config_solar_plain2_BRATS_HGG.json --fold 1 --seed $SEED