import pandas as pd
from tqdm import tqdm

dataset = pd.read_csv("*.csv", sep=";")

lexicon = pd.read_csv("sentiment_lexicon.csv", sep=";")

n = len(dataset.columns)
dataset.insert(loc=n, column="article_sentiment", value="0", allow_duplicates=True)
n = len(dataset.columns)
dataset.insert(loc=n, column="n_of_lexicon_words", value="0", allow_duplicates=True)

# use a for - loop to go through all articles
for i in tqdm(dataset.index):
    article = dataset.loc[i].at["text"]

    # convert article to list
    article = article.split(" ")

    # set starting values for word-counts and sentiment
    article_sentiment = 0
    n_of_countable_words = 0

    # compare article to lexicon
    for j in range(len(article)):
        sf = lexicon.index[lexicon['words'] == article[j]]
        for m in range(len(sf)):
            n_of_countable_words = n_of_countable_words + 1
            article_sentiment = article_sentiment + lexicon.loc[sf[m]].at['value']
            break
    dataset.at[i, 'article_sentiment'] = article_sentiment
    dataset.at[i, 'n_of_lexicon_words'] = n_of_countable_words

# save datasheet to csv
dataset.to_csv('*.csv', sep=';')