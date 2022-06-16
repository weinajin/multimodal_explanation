#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-07:00:00
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/labname-lab/authorid/shortcut/log/tumorsyn_gethm/s4_%A.out
#SBATCH --job-name s4_hm
#SBATCH --partition=long

FOLD=4
module load LANG/PYTHON/3.7.6
source /home/authorid/brain/bin/activate

cd /project/labname-lab/authorid/shortcut/BRATS_IDH/code/
srun python xai_pipeline.py --ts solar --config sh/tumorsyn_shortcut_xai_solar/tumorsyn_shortcut_xai_solar.json --fold $FOLD --bs 1 --job gethm -r /project/labname-lab/authorid/shortcut/log/tumorsyn_gethm/tumorsyn_shortcut/1123_115902_fold_4
