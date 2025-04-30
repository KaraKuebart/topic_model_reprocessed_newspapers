#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --partition=lm_long
#SBATCH --mem=1800G
#SBATCH --job-name=sent_a
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=nkuebart@uni-bonn.de
#SBATCH --mail-type=ALL
#SBATCH --time 1-12:00:00



ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_gensim
export PYTHONPATH=${PYTHONPATH}:
python src/topic_rnews/preparation_and_sentiment_analysis.py -n-cpu 64 -d /home/nkuebart_hpc/npzs/ -imp [0, 5000] -od output/speedtest_df_raw.csv -out output/speedtest_df
python src/topic_rnews/topic_model_gensim.py
python src/topic_rnews/topic_model_bert.py

conda deactivate
conda activate for_bert
python src/topic_rnews/topic_model_bert.py
python src/topic_rnews/topic_model_tomoto.py