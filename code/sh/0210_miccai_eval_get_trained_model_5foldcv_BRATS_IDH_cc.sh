#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=6
#SBATCH --array=1-5
#SBATCH --time=5-07:00:00
#SBATCH --account=rrg-PIID
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --output=../out/0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_%A_%a.out
#SBATCH --job-name 0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc
python train.py --config sh/0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc.json --fold=$SLURM_ARRAY_TASK_ID
