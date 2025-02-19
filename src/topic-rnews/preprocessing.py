import string

import pandas as pd

from tqdm import tqdm
from resources.pythonic_resources import stopwords, consolidations, lemmata

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def main(dataset:pd.DataFrame) -> pd.DataFrame:

    dataset['text'] = dataset['text'].str.lower()

    for i in tqdm(consolidations.index):
        dataset = dataset.replace(to_replace=consolidations.loc[i].at["letters"],
                                  value=consolidations.loc[i].at["replace"],
                                  regex=True)

    for j in tqdm(lemmata.index):
        dataset = dataset.replace(to_replace=" {} ".format(lemmata.loc[j].at["word"]),
                                  value=" {} ".format(lemmata.loc[j].at["replace"]), regex=True)

    for k in tqdm(stopwords):
        dataset = dataset.replace(to_replace=f" {k} ", value=" ", regex=True)

    # ToDo make sure the new punctuation removal does what it should
    dataset['text'] = dataset['text'].apply(remove_punctuation)

    # old punctuation method:
    # punctuation = ''', : ; - _ # ' ~ ` ´ = / & % $ § " ! ° ^ < > | '''
    # punctuation = punctuation.split(' ')
    # for sign in punctuation:
    #     dataset['text'] = dataset['text'].str.replace('{}'.format(sign), '', regex=True)

    return dataset

