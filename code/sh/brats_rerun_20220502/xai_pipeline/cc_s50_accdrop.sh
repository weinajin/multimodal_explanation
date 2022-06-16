#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=rrg-PIID
#SBATCH --time=12:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/authorid/results_brats_rerun/heatmaps/log/%x_%j_%N.out
#SBATCH --error=/scratch/authorid/results_brats_rerun/heatmaps/log/%x_%j_%N.err
#SBATCH --job-name s50-rd_accdrop

module load python/3.7
module load openslide

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip

pip install -r /scratch/authorid/BRATS_IDH/code/requirement_cc.txt

cd /scratch/authorid/BRATS_IDH/code/
git pull

SEED=50

python xai_pipeline.py --config sh/xai_cc_plain2_BRATS_HGG.json --fold 1 --seed $SEED --bs 1 --ts cc --job acc_drop -r /scratch/authorid/results_brats_rerun/heatmaps/seed_50/ |& tee -a /scratch/authorid/results_brats_rerun/heatmaps/log/s50-rd_accdrop.txt
