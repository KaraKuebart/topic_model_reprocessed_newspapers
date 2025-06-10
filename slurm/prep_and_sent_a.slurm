#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --partition=lm_long
#SBATCH --mem=500G
#SBATCH --job-name=sentiment_analysis
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL
#SBATCH --time 7-00:00:00

ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_gensim

export PYTHONPATH=${PYTHONPATH}:
python src/topic_rnews/preparation_and_sentiment_analysis.py -n-cpu 96 -d /home/nkuebart_hpc/npzs/ -od output/third_raw.csv -out output/third