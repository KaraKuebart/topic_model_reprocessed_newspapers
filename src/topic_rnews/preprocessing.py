import string

import pandas as pd

from tqdm import tqdm
from resources.pythonic_resources import stopwords, consolidations, lemmata

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def drop_short_lines(dataframe):
    print('before_reduction:', len(dataframe))
    for j in tqdm(dataframe.index):
        text = dataframe.loc[j]['text']
        if not isinstance(text, str) or len(text) < 50:
            dataframe.drop(j, inplace=True)
    print('after reduction:', len(dataframe))
    return dataframe

def main(dataset:pd.DataFrame) -> pd.DataFrame:

    dataset['text'] = dataset['text'].str.lower()

    print('applying consolidations')

    for i in tqdm(consolidations.index):
        dataset = dataset.replace(to_replace=consolidations.loc[i, "letters"],
                                  value=consolidations.loc[i, "replace"],
                                  regex=True)

    print('applying lemmata')

    for j in tqdm(lemmata.index):
        dataset = dataset.replace(to_replace=" {} ".format(lemmata.loc[j].at["word"]),
                                  value=" {} ".format(lemmata.loc[j].at["replace"]), regex=True)

    print('applying stopwords')

    for k in tqdm(stopwords):
        dataset.replace(to_replace=f" {k} ", value=" ", regex=True, inplace=True)

    dataset.replace(to_replace="-\n", value="", regex=True, inplace=True)
    dataset.replace(to_replace="\n", value=" ", regex=True, inplace=True)

    print('removing punctuation')

    # ToDo make sure the new punctuation removal does what it should
    dataset['text'] = dataset['text'].apply(remove_punctuation)

    # old punctuation method:
    # punctuation = ''', : ; - _ # ' ~ ` ´ = / & % $ § " ! ° ^ < > | '''
    # punctuation = punctuation.split(' ')
    # for sign in punctuation:
    #     dataset['text'] = dataset['text'].str.replace('{}'.format(sign), '', regex=True)

    return dataset

