#!/bin/bash 
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/BRATS_IDH/log/0423_%A_BRATS_HGG_fold1.out
#SBATCH --job-name 0423_BRATS_HGG_cc
#SBATCH --partition=long
#SBATCH --nodelist=cs-venus-04

module load LANG/PYTHON/3.7.6
source /home/authorid/brain/bin/activate

cd /project/labname-lab/authorid/BRATS_IDH/code/
srun python train.py --config sh/config_solar_plain2_BRATS_HGG.json --fold 2
