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
#SBATCH --job-name s20-gethm

module load python/3.7
module load openslide

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip

pip install -r /scratch/authorid/BRATS_IDH/code/requirement_cc.txt

cd /scratch/authorid/BRATS_IDH/code/
git pull

SEED=20

python xai_pipeline.py --config sh/xai_cc_plain2_BRATS_HGG.json --fold 1 --seed $SEED --bs 1 --ts cc --job gethm -r /scratch/authorid/brats_rerun/BRATS_HGG/0506_150529_fold_1_seed_20/
