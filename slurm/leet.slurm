#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --partition=lm_short
#SBATCH --mem=500G
#SBATCH --job-name=leet
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL
#SBATCH --time 08:00:00

ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_gensim

export PYTHONPATH=${PYTHONPATH}:
python src/topic_rnews/topic_model_leet.py -ldf output/12th_pre.csv -leet 0.05 -out output/leet_12th
