import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

def get_processing_docs(consolidations_path:str='../resources/consolidations.csv', lemmas_path:str='../resources/lemmas.csv', stopwords_path:str='../resources/stopwords.csv') -> \
tuple[DataFrame, DataFrame, DataFrame]:
    consolidations = pd.read_csv(consolidations_path, sep=";", dtype="string")
    lemmas = pd.read_csv(lemmas_path, sep=";", dtype="string")
    stopwords = pd.read_csv(stopwords_path, sep=";", dtype="string")
    return consolidations, lemmas, stopwords

def preprocess(dataset:pd.DataFrame) -> pd.DataFrame:
    consolidations, lemmas, stopwords = get_processing_docs()
    dataset['text'] = dataset['text'].str.lower()

    for i in tqdm(consolidations.index):
        dataset = dataset.replace(to_replace=consolidations.loc[i].at["letters"],
                                  value=consolidations.loc[i].at["replace"],
                                  regex=True)

    for j in tqdm(lemmas.index):
        dataset = dataset.replace(to_replace=" {} ".format(lemmas.loc[j].at["word"]),
                                  value=" {} ".format(lemmas.loc[j].at["replace"]), regex=True)

    for k in tqdm(stopwords.index):
        dataset = dataset.replace(to_replace=" {} ".format(stopwords.loc[k].at["stopwords"]), value=" ", regex=True)

    punctuation = ''', : ; - _ # ' ~ ` ´ = / & % $ § " ! ° ^ < > | '''
    punctuation = punctuation.split(' ')
    for sign in punctuation:
        dataset['text'] = dataset['text'].str.replace('{}'.format(sign), '', regex=True)

    return dataset

