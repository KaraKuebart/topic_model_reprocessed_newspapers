#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=sgpu_short
#SBATCH --mem=400G
#SBATCH --job-name=bertopic_raw
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL

ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_bert
python src/topic_rnews/topic_model_bertopic_from_raw.py -ldf output/6th_raw.csv -out output/bert_6th_raw
