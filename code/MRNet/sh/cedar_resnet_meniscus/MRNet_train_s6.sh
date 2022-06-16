#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/authorid/shortcut/log/MRNet/meniscus/m6_1119_mrnet_%A.out
#SBATCH --job-name m6_mrnet
#SBATCH --account=rrg-PIID


module load python/3.7
module load openslide

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip

pip install -r /scratch/authorid/BRATS_IDH/code/requirement.txt

cd /scratch/authorid/shortcut/MRNet/
python train.py --rundir /scratch/authorid/shortcut/log/MRNet/meniscus/seed6 --task meniscus --backbone resnet18 --seed 6
