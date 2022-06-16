#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/shortcut/log/MRNet/s1_1118_mrnet_%A.out
#SBATCH --job-name s1_mrnet
#SBATCH --partition=long

module load LANG/PYTHON/3.7.6
source /home/authorid/brain/bin/activate

cd /project/labname-lab/authorid/shortcut/BRATS_IDH/MRNet
srun python train.py --rundir /project/labname-lab/authorid/shortcut/log/MRNet/seed1 --task acl --seed 1
