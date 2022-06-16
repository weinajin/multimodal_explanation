#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/tumorsyn_xai/log/tumorsyn/gethm_fold_7/0531_tumorsyn_%A.out
#SBATCH --job-name 0531_xai_svs
#SBATCH --partition=long

module load LANG/PYTHON/3.7.6
source /home/authorid/brain/bin/activate

cd /project/labname-lab/authorid/tumorsyn_xai/code/
srun python xai_pipeline.py --ts solar --config sh/tumorsyn_xai_ts.json --fold 7 -r /project/labname-lab/authorid/tumorsyn_xai/log/tumorsyn/gethm_fold_7 --bs 1
