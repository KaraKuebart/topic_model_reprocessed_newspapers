import string
import time

import pandas as pd
import datetime
from tqdm import tqdm


from resources.pythonic_resources import stopwords, consolidations, lemmata

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def drop_short_lines(dataframe):
    print(datetime.datetime.now(), ': dropping short lines. Length before_reduction:', len(dataframe))
    dataframe['text'] = dataframe['text'].astype(str)
    dataframe = dataframe.drop(dataframe[dataframe['text'].str.len() < 50].index)
    print(datetime.datetime.now(), ': Length after reduction:', len(dataframe))
    return dataframe

def main(dataset:pd.DataFrame) -> pd.DataFrame:
    dataset['text'] = dataset['text'].str.lower()
    dataset = dataset.p_replace(to_replace="-\n", value="", regex=True)
    dataset = dataset.p_replace(to_replace="\n", value=" ", regex=True)


    print(datetime.datetime.now(), ': applying consolidations')
    a = time.time()
    dataset = dataset.replace(consolidations, regex=True) # works faster in normal pandas than parallel pandas
    b = time.time()

    # for i in tqdm(consolidations.index):
    #    dataset = dataset.p_replace(to_replace=consolidations.loc[i, "letters"],
                                  #value=consolidations.loc[i, "replace"],
                                  #regex=True)

    print(datetime.datetime.now(), ': applying lemmata')
    dataset = dataset.replace(lemmata, regex=True)

    #for j in tqdm(lemmata.index):
    #    dataset = dataset.p_replace(to_replace=f""" {lemmata.loc[j].at["word"]} """,
                                  #value=f""" {lemmata.loc[j].at["replace"]} """, regex=True)

    print(datetime.datetime.now(), ': applying stopwords')
    dataset = dataset.replace(stopwords, regex=True)
    #for k in tqdm(stopwords):
    #    dataset = dataset.p_replace(to_replace=f" {k} ", value=" ", regex=True)



    print(datetime.datetime.now(), ': removing punctuation')
    dataset['text'] = dataset['text'].p_apply(remove_punctuation)

    print(datetime.datetime.now(), ': preprocessing done')
    return dataset
