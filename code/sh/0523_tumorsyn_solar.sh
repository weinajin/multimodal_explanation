#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/shortcut/log/1118_tumorsyn_%A.out
#SBATCH --job-name save_tumorsyn
#SBATCH --partition=long
#SBATCH --nodelist=cs-venus-04

module load LANG/PYTHON/3.7.6
source /home/authorid/brain/bin/activate

cd /project/labname-lab/authorid/shortcut/BRATS_IDH/code/
srun python train.py --config sh/tumorsyn_solar.json --save presaved --seed 2
