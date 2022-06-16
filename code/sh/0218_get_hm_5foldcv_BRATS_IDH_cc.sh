#!/bin/bash 
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-5%1
#SBATCH --time=5-07:00:00
#SBATCH --account=rrg-PIID
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=ALL
#SBATCH --error=../out/0303_%A_error_%a.out
#SBATCH --output=../out/0303_%A_get_hm_5foldcv_BRATS_IDH_cc_%a.out
#SBATCH --job-name 0224_get_hm_5foldcv_BRATS_IDH_cc
python xai_pipeline.py --config sh/0218_get_hm_5foldcv_BRATS_IDH_cc.json --fold $SLURM_ARRAY_TASK_ID --bs 1 -r ../exp_log/xai_exp/get_hm/fold_$SLURM_ARRAY_TASK_ID
