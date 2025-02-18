import pandas as pd
from tqdm import tqdm
import argparse

import read_data


if __name__ == "__main__":

    # import data from numpy arrays
    news_df = read_data.create_dataframe(read_data.get_args())


    # ToDo enter here code to add headings and corresponding paragraphs together.
    #  Sorting the df might be necessary.

    # ToDo do some general preprocessing: remove long s, create stopword list...
    # figure out whether to apply stopwords to all analyses...


    # drop unuseful data
    for i in tqdm(news_df.index):
        if type(news_df.loc[i, "text"]) != str or len(news_df.loc[i, "text"]) < 50:
            news_df.drop(i, inplace=True)










    news_df.to_csv('test_batch_pre.csv', sep=';')