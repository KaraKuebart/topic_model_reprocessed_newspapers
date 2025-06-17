#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --partition=lm_long
#SBATCH --mem=300G
#SBATCH --job-name=gensim
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL

ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_gensim

export PYTHONPATH=${PYTHONPATH}:
python src/topic_rnews/topic_model_gensim.py -ldf output/12th_pre.csv -out output/gensim_12th
