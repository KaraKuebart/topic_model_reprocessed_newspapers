#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --partition=intelsr
#SBATCH --mem=250G
#SBATCH --job-name=sent_a
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL
#SBATCH --time 2-12:00:00

ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_gensim

export PYTHONPATH=${PYTHONPATH}:
python src/topic_rnews/topic_model_tomoto.py -ldf output/prep_pre.csv -out output/tomoto
