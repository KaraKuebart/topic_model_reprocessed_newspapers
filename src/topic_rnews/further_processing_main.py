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


def call_leet_topic(df: pd.DataFrame, topic_number) -> pd.DataFrame:
    df_red = df.loc[df['leet_labels'] == topic_number]
    df_red = run_leet_topic(df_red, 0.1)
    df_red.to_csv(f'output/all_double_topic_on_topicnr_{topic_number}.csv', sep=';', index=False)
    print(datetime.datetime.now(), ': finished topic nr.', topic_number)
    return df_red

if __name__ == "__main__":
#    print(datetime.datetime.now(), 'beginning')
#
    args = read_data.get_args()
#
#    # import data
#    b1: DataFrame = import_df('batch_1')
#    b2: DataFrame = import_df('batch_2')
#    b3: DataFrame = import_df('batch_3')
#    b4: DataFrame = import_df('batch_4')
#    b5: DataFrame = import_df('batch_5')
#    news_df = pd.concat([b1, b2, b3, b4, b5], ignore_index=True)
#    # save all
#    news_df.to_csv('output/news_df_full.csv', sep=';', index=False)
#
#    # topic modelling
#    news_df = run_leet_topic(news_df, args.leet_distance)
#    print(datetime.datetime.now(), ": finished leet topic model, saving, then starting LDA")
#    news_df.to_csv(f'output/news_df_lda_{args.leet_distance}.csv', sep=';', index=False)
#
#    # export results
#    news_df.to_csv(args.output_document_path + '.csv', sep=';', index=False)
#
#    # double leet:
#    leet_df0 = run_leet_topic(news_df, 0.1)
#    del news_df
#    leet_df0.to_csv(args.output_document_path + 'rough_leet_0.5.csv', sep=';', index=False)
#    topics = set(leet_df0['leet_labels'].tolist())
#    print(datetime.datetime.now(), f": finished leet topic, now starting second leet topic on all {len(topics)} topics: {topics}")
#    df_list = []
#    for topic in topics:
#        df = call_leet_topic(leet_df0, topic)
#        df_list.append(df)
#    pd.concat(df_list).to_csv(args.output_document_path + 'double_leet.csv', sep=';', index=False)

    news_df = import_df(args.batch_nr)
    news_df = run_leet_topic(news_df, args.leet_distance)
    news_df.to_csv(f'output/leet_results{args.batch_nr}.csv', sep=';', index=False)
