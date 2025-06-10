#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=sgpu_long
#SBATCH --mem=400G
#SBATCH --job-name=bertopic
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL
#SBATCH --time 5-00:00:00

ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_bert
python src/topic_rnews/topic_model_bert.py -ldf output/third_pre.csv -leet 0.2 -out output/bert_third
