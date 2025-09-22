#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=sgpu_medium
#SBATCH --mem=800G
#SBATCH --job-name=analysis_speedtest
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out
#SBATCH --mail-user=[enter your own email here]
#SBATCH --mail-type=ALL
#SBATCH --time 18:00:00



ln -sf logs/job.$SLURM_JOB_ID.err latest.err
ln -sf logs/job.$SLURM_JOB_ID.out latest.out

cd /home/nkuebart_hpc/topic_model_reprocessed_newspapers/
eval "$(conda shell.bash hook)"

conda activate for_gensim
export PYTHONPATH=${PYTHONPATH}:
python src/topic_rnews/preparation_and_sentiment_analysis.py -n-cpu 64 -d /home/nkuebart_hpc/npzs/ -imp 0 5000 -od output/speedtest_df_raw.csv -out output/speedtest_df
python src/topic_rnews/topic_model_gensim.py -ldf output/speedtest_df_pre.csv -lda 500 -out output/speedtest_df
python src/topic_rnews/topic_model_leet.py -ldf output/speedtest_df_pre.csv -leet 0.2 -out output/speedtest_df
python src/topic_rnews/topic_model_tomoto.py -ldf output/speedtest_df_pre.csv -leet 0.2 -out output/speedtest_df

conda deactivate
conda activate for_bert
python src/topic_rnews/topic_model_bert.py -ldf output/speedtest_df_pre.csv -leet 0.2 -out output/speedtest_df
