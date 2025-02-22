import string
import time
import pandas as pd

from tqdm import tqdm
from resources.pythonic_resources import stopwords, consolidations, lemmata

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def drop_short_lines(dataframe):
    print(time.time(), ': dropping short lines. Length before_reduction:', len(dataframe))
    dataframe['text'] = dataframe['text'].astype(str)
    dataframe = dataframe.drop(dataframe[dataframe['text'].str.len() < 50].index)
    print(time.time(), ': Length after reduction:', len(dataframe))
    return dataframe

def main(dataset:pd.DataFrame) -> pd.DataFrame:

    print(time.time(), ': applying consolidations')
    for i in tqdm(consolidations.index):
        dataset = dataset.p_replace(to_replace=consolidations.loc[i, "letters"],
                                  value=consolidations.loc[i, "replace"],
                                  regex=True)

    print(time.time(), ': applying lemmata')
    for j in tqdm(lemmata.index):
        dataset = dataset.p_replace(to_replace=f""" {lemmata.loc[j].at["word"]} """,
                                  value=f""" {lemmata.loc[j].at["replace"]} """, regex=True)

    print(time.time(), ': applying stopwords')
    for k in tqdm(stopwords):
        dataset.p_replace(to_replace=f" {k} ", value=" ", regex=True, inplace=True)

    dataset.p_replace(to_replace="-\n", value="", regex=True, inplace=True)
    dataset.p_replace(to_replace="\n", value=" ", regex=True, inplace=True)

    print(time.time(), ': removing punctuation')
    dataset['text'] = dataset['text'].p_apply(remove_punctuation)

    print(time.time(), ': preprocessing done')
    return dataset
