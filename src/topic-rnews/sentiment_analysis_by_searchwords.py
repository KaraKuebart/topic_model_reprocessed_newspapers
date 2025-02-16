import pandas as pd
from tqdm import tqdm

dataset = pd.read_csv("*.csv", sep=";")

lexicon = pd.read_csv("sentiment_lexicon.csv", sep=";")

analysis_word = ""

range_ =

lexicon['words'] = lexicon['words'].astype(str)

# use a for - loop to go through all articles
for i in tqdm(dataset.index):
    article = str(dataset.loc[i].at["text"])

    # convert article to list
    article = article.split(" ")

    # set starting values for word-counts and sentiment
    article_sentiment = 0
    n_of_analysis_words = 0
    n_of_countable_words = 0
    found_words = []

    # search for the analysis_word in the list
    for j in range(len(article)):
        if article[j] == analysis_word:
            n_of_analysis_words = n_of_analysis_words + 1

            # make a list of all words in the range, and put it in a String with | separator
            range_list = []
            for k in range((j - range_), (j + range_ + 1)):
                if k >= (len(article)):
                    break
                if k < 0:
                    continue
                else:
                    range_list.append(article[k])
                    # compare your rangelist to the lexicon
            for k in range(len(range_list)):
                sf = lexicon.index[lexicon['words'] == range_list[k]]

                for m in range(len(sf)):
                    n_of_countable_words = n_of_countable_words + 1
                    article_sentiment = article_sentiment + lexicon.loc[sf[m]].at['value']
                    found_words.append(lexicon.loc[sf[m]].at['words'])

                    break
        else:
            continue
    dataset.at[i, 'article_sentiment'] = article_sentiment
    dataset.at[i, 'n_of_analysis_word'] = n_of_analysis_words
    dataset.at[i, 'n_of_lexicon_words'] = n_of_countable_words
    dataset.at[i, 'lexicon_words'] = ' '.join(found_words)

# save datasheet to csv
dataset.to_csv('*.csv', sep=';')