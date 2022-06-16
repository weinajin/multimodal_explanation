#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/shortcut/log/MRNet/mi_log/s8_%A.out
#SBATCH --job-name s8_mi
#SBATCH --partition=long

module load LANG/PYTHON/3.7.6
source /home/authorid/brain/bin/activate

SEED=8

cd /project/labname-lab/authorid/shortcut/BRATS_IDH/code
python xai_mrnet.py --machine solar -c /project/labname-lab/authorid/shortcut/BRATS_IDH/code/sh/mrnet_gethm/mrnet_xai_solar.json --model_path /project/labname-lab/authorid/ --seed $SEED --job mi_readhm