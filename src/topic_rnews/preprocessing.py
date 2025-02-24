import string
from tqdm import tqdm

import pandas as pd
import datetime


from resources.pythonic_resources import stopwords, consolidations, lemmata


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def drop_short_lines(dataframe):
    print(datetime.datetime.now(), ': dropping short lines. Length before_reduction:', len(dataframe))
    dataframe['text'] = dataframe['text'].astype(str)
    dataframe = dataframe.drop(dataframe[dataframe['text'].str.len() < 50].index)
    print(datetime.datetime.now(), ': Length after reduction:', len(dataframe))
    return dataframe


def join_headings_w_paragraphs(local_df: pd.DataFrame) -> pd.DataFrame:
    indices = local_df.index
    for i in tqdm(indices[:-2]):
        if local_df.loc[i]['class'] == 'heading' and local_df.loc[i + 1]['class'] == 'paragraph' and float(
                local_df.loc[i]['confidence']) > 0.5 and float(local_df.loc[i + 1]['confidence']) > 0.5 and \
                local_df.loc[i]['region'] == int(local_df.loc[i + 1]['region']) - 1:
            local_df.loc[i]['class'] = 'joined'
            local_df.loc[i]['confidence'] = (local_df.loc[i]['confidence'] + local_df.loc[i + 1]['confidence']) / 2.0
            local_df.loc[i]['text'] = str(local_df.loc[i]['text']) + ' ' + str(local_df.loc[i + 1]['text'])
            local_df.drop(local_df.loc[i + 1], inplace=True)
    print(datetime.datetime.now(), 'headings and paragraphs joined')
    return local_df


def main(dataset:pd.DataFrame) -> pd.DataFrame:
    dataset['text'] = dataset['text'].str.lower()
    dataset = dataset.p_replace(to_replace="-\n", value="", regex=True)
    dataset = dataset.p_replace(to_replace="\n", value=" ", regex=True)

    print(datetime.datetime.now(), ': applying consolidations')
    dataset = dataset.replace(consolidations, regex=True) # works faster in normal pandas than parallel pandas

    print(datetime.datetime.now(), ': applying lemmata')
    dataset = dataset.replace(lemmata, regex=True)

    print(datetime.datetime.now(), ': applying stopwords')
    dataset = dataset.replace(stopwords, regex=True)

    print(datetime.datetime.now(), ': removing punctuation')
    dataset['text'] = dataset['text'].p_apply(remove_punctuation)

    print(datetime.datetime.now(), ': preprocessing done')
    return dataset
