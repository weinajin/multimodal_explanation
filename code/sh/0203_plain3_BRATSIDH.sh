#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=6
#SBATCH --time=5-07:00:00
#SBATCH --account=rrg-PIID
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=../out/0203_plain3_BRATSIDH.out
#SBATCH --job-name 0203_plain_BRATSIDH
python train.py --config sh/config_cc_plain3_BRATS_IDH.json --fold=1
