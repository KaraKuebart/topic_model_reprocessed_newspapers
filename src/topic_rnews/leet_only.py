from pandas import DataFrame

import read_data
import preprocessing
import sentiment_analysis_fast
from topic_model import run_lda, run_leet_topic
# from parallel_pandas import ParallelPandas
import datetime
import pandas as pd
from leet_topic import leet_topic

def import_df(name:str) -> pd.DataFrame:
    return pd.read_csv(f'output/{name}_pre.csv')


def call_leet_topic(df: DataFrame, topic_number) -> pd.DataFrame:
    df_red = df.loc[df['leet_labels'] == topic_number]
    df_red = run_leet_topic(df_red, 0.1)
    df_red.to_csv(f'output/all_double_topic_on_topicnr_{topic_number}.csv', sep=';', index=False)
    print(datetime.datetime.now(), ': finished topic nr.', topic_number)
    return df_red

if __name__ == "__main__":
    args = read_data.get_args()

    news_df = import_df(args.batch_nr)
    news_df = run_leet_topic(news_df, args.leet_distance)
    news_df.to_csv(f'output/leet_results{args.batch_nr}.csv', sep=';', index=False)
