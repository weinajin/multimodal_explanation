#!/bin/bash 
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-07:00:00
#SBATCH --account=rrg-PIID
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=../out/0421_%A_BRATS_HGG_fold1.out
#SBATCH --job-name 0421_BRATS_HGG_cc

module load python/3.7
module load openslide

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip

pip install -r /scratch/authorid/BRATS_IDH/code/requirement.txt

cd /scratch/authorid/BRATS_IDH/code/
python train.py --config sh/config_cc_plain2_BRATS_HGG.json --fold 1
