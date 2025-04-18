import pandas as pd
import datetime
import time

lexicon = pd.read_csv("resources/sentiment_lexicon.csv", sep=";")
lexicon['words'] = lexicon['words'].astype(str)


def article_row(article:str) -> (float, int):
    # convert article to list
    article = article.split(" ")

    # set starting values for word-counts and sentiment
    article_sentiment = 0
    n_of_countable_words = 0

    # compare article to lexicon
    for word in article:
        sf = lexicon.index[lexicon['words'] == word]
        for m in range(len(sf)):
            n_of_countable_words = n_of_countable_words + 1
            article_sentiment = article_sentiment + lexicon.loc[sf[m]].at['value']
            break
    return article_sentiment, n_of_countable_words


def split_results(dataframe:pd.DataFrame) -> pd.DataFrame:
    dataframe['sentiment_values'] = dataframe['mixed_results'].str[0]
    dataframe['n_of_countable_words'] = dataframe['mixed_results'].str[1]
    return dataframe

def per_article(local_df: pd.DataFrame):
    a = time.time()
    print(datetime.datetime.now(), ': calculating sentiment scores per article')
    local_df['mixed_results'] = local_df['text'].p_apply(article_row)
    local_df = split_results(local_df)
    b = time.time()
    print(datetime.datetime.now(), f': calculating sentiment scores per article took {b-a} seconds')
    local_df.drop(columns=['mixed_results'], inplace=True)
    return local_df



def words_row(row: pd.Series, analysis_words:list, analysis_range:tuple=(0, 30)) -> pd.Series:
    article = row['text']

    # convert article to list
    article = article.split(" ")

    # set starting values for word-counts and sentiment
    article_sentiments = {}
    found_analysis_words = {}
    n_of_countable_words = {}
    found_words = {}
    # TODO: split found_words into positive and negative
    for word in analysis_words:
        found_analysis_words[word] = 0
        article_sentiments[word] = 0
        n_of_countable_words[word] = 0
        found_words[word] = []

    # search for the analysis_word in the list
    for j in range(len(article)):
        if article[j] in analysis_words:
            current_word = article[j]
            found_analysis_words[current_word] += 1

            # make a list of all words in the range, and put it in a String with | separator
            range_list = []
            for k in range((j - analysis_range[0]), (j + analysis_range[1] + 1)):
                if k >= (len(article)):
                    break
                if k < 0:
                    continue
                range_list.append(article[k])
                    # compare your rangelist to the lexicon
            for k in range(len(range_list)):
                sf = lexicon.index[lexicon['words'] == range_list[k]]

                for m in range(len(sf)):
                    n_of_countable_words[current_word] += 1
                    article_sentiments[current_word] += lexicon.loc[sf[m]].at['value']
                    found_words[current_word].append(lexicon.loc[sf[m]].at['words'])
                    break
        else:
            continue
    for word in analysis_words:
        row[f'article_sentiment_for:_{word}'] = article_sentiments[word]
        row[f'n_of_analysis_word_for:_{word}'] = found_analysis_words[word]
        row[f'n_of_lexicon_words_for:_{word}'] = n_of_countable_words[word]
        row[f'lexicon_words_for:_{word}'] = ' '.join(found_words[word])
    return row


def by_words(dataset: pd.DataFrame, analysis_words:list, analysis_range:tuple=(0, 30)) -> pd.DataFrame:
    a = time.time()
    print(datetime.datetime.now(), ': calculating sentiment scores by words')
    dataset = dataset.p_apply(words_row, axis=1, args=(analysis_words, analysis_range))
    b = time.time()
    print(datetime.datetime.now(), f': calculating sentiment scores by words took {b-a} seconds')
    return dataset
