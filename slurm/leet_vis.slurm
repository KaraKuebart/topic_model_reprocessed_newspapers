#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --partition=lm_short
#SBATCH --mem=1800G
#SBATCH --job-name=leet_vis
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL
#SBATCH --time 01:25:00



ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"
conda activate base
export PYTHONPATH=${PYTHONPATH}:
python src/topic_rnews/visualization_leet_b1.py
