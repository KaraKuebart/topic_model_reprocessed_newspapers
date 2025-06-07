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
    dataframe = dataframe.drop(dataframe[dataframe['text'].str.len() < 15].index)
    print(datetime.datetime.now(), ': Length after reduction:', len(dataframe))
    return dataframe


def join_headings_w_paragraphs(local_df: pd.DataFrame) -> pd.DataFrame:
    indices = local_df.index
    for i in tqdm(indices[:-2]):
        if i in local_df.index:
            if local_df.at[i, 'class'] == 'heading' and local_df.at[i + 1, 'class'] == 'paragraph' and float(
                    local_df.at[i, 'confidence']) > 0.5 and float(local_df.at[i + 1, 'confidence']) > 0.5 and \
                    int(local_df.at[i, 'region']) == int(local_df.at[i + 1, 'region']) - 1:
                local_df.at[i, 'class'] = 'joined'
                local_df.at[i, 'confidence'] = (float(local_df.at[i, 'confidence']) + float(local_df.at[i + 1, 'confidence'])) / 2.0
                local_df.at[i, 'text'] = str(local_df.at[i, 'text']) + ' ' + str(local_df.at[i + 1, 'text'])
                local_df.drop(i + 1, inplace=True)
    print(datetime.datetime.now(), 'headings and paragraphs joined')
    return local_df


def main(dataset:pd.DataFrame) -> pd.DataFrame:
    print(datetime.datetime.now(), ': beginning preprocessing main function. Making lowercase ...')
    dataset['text'] = dataset['text'].str.lower()
    print(datetime.datetime.now(), ': removing newlines with dash')
    dataset = dataset.p_replace(to_replace="-\n", value="", regex=True)
    print(datetime.datetime.now(), ': removing newlines')
    dataset = dataset.p_replace(to_replace="\n", value=" ", regex=True)

    print(datetime.datetime.now(), ': applying consolidations')

    dataset = dataset.p_replace(consolidations, regex=True) # works faster in normal pandas than parallel pandas in most cases

    print(datetime.datetime.now(), ': applying lemmata')

    dataset = dataset.p_replace(lemmata, regex=True)

    print(datetime.datetime.now(), ': applying stopwords')

    dataset = dataset.p_replace(stopwords, regex=True)

    print(datetime.datetime.now(), ': removing punctuation')

    dataset['text'] = dataset['text'].p_apply(remove_punctuation)

    print(datetime.datetime.now(), ': preprocessing done')
    return dataset
