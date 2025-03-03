import pandas as pd
import sentiment_analysis_fast
from parallel_pandas import ParallelPandas


if __name__ == "__main__":

    ParallelPandas.initialize(n_cpu=192, split_factor=8)

    news_df = pd.read_csv('output/news_df_full.csv', sep=';')
    news_df['text'] = news_df['text'].astype(str)
    news_df = sentiment_analysis_fast.per_article(news_df)

    words_of_interest = ['estland', 'lettland', 'livland', 'litauen', 'finnland', 'england', 'schweden', 'norwegen',
                         'daenemark', 'frankreich']
    range_tuple = (0, 30)
    news_df = sentiment_analysis_fast.by_words(news_df, words_of_interest, range_tuple)
    news_df.to_csv(f'output/sentiment_results.csv', sep=';', index=False)
