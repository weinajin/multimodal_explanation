#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --account=rrg-PIID
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/authorid/brats_rerun/%x_%N_%j.out
#SBATCH --error=/scratch/authorid/brats_rerun/%x_%N_%j.err
#SBATCH --job-name s50-BratsTrain

module load python/3.7
module load openslide

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip

pip install -r /scratch/authorid/BRATS_IDH/code/requirement_cc.txt

cd /scratch/authorid/BRATS_IDH/code/


SEED=50

python train.py --config sh/config_cc_plain2_BRATS_HGG.json --fold 1 --seed $SEED
