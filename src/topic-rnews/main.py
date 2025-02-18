import pandas as pd
from tqdm import tqdm
import nltk

import read_data
from resources.pythonic_resources import stopwords
from preprocessing_for_sentiment_analysis

if __name__ == "__main__":
    news_df = read_data.create_dataframe(read_data.get_args())

    # ToDo enter here code to add headings and corresponding paragraphs together.
    #  Sorting the df might be necessary.

    # ToDo do some general preprocessing: remove long s, create stopword list...
    #  figure out whether to apply stopwords to all analyses...
    for stopword in stopwords:
        if f' {stopword} ' in text:
            text = text.replace(f' {stopword} ', ' ')
    # import data from numpy arrays

    # drop unuseful data
    print(len(news_df))

    for i in tqdm(news_df.index):
        if type(news_df.loc[i, "text"]) != str or len(news_df.loc[i, "text"]) < 50:
            news_df.drop(i, inplace=True)
    print(len(news_df))


    news_df.to_csv('temp.csv', sep=';')