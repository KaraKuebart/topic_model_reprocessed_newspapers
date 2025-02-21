import pandas as pd
from tqdm import tqdm

lexicon = pd.read_csv("resources/sentiment_lexicon.csv", sep=";")
lexicon['words'] = lexicon['words'].astype(str)


def per_article(dataset: pd.DataFrame):
    print('calculating sentiment scores per article')
    for i in tqdm(dataset.index):
        article = dataset.loc[i].at["text"]

        # convert article to list
        article = article.split(" ")

        # set starting values for word-counts and sentiment
        article_sentiment = 0
        n_of_countable_words = 0

        # compare article to lexicon
        for j in enumerate(article):
            sf = lexicon.index[lexicon['words'] == article[j]]
            for m in enumerate(sf):
                n_of_countable_words = n_of_countable_words + 1
                article_sentiment = article_sentiment + lexicon.loc[sf[m]].at['value']
                break
        dataset.at[i, 'article_sentiment'] = article_sentiment
        dataset.at[i, 'n_of_lexicon_words'] = n_of_countable_words
    return dataset




def by_words(dataset: pd.DataFrame, analysis_words:list, analysis_range:tuple=(0, 30)) -> pd.DataFrame:
    print('calculating sentiment scores by words')
    for i in tqdm(dataset.index):
        article = str(dataset.loc[i].at["text"])

        # convert article to list
        article = article.split(" ")

        # set starting values for word-counts and sentiment
        article_sentiments = {}
        found_analysis_words = {}
        n_of_countable_words = {}
        found_words = {}
        for word in analysis_words:
            found_analysis_words[word] = 0
            article_sentiments[word] = 0
            n_of_countable_words[word] = 0
            found_words[word] = []

        # search for the analysis_word in the list
        for j in enumerate(article):
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
                for k in enumerate(range_list):
                    sf = lexicon.index[lexicon['words'] == range_list[k]]

                    for m in enumerate(sf):
                        n_of_countable_words[current_word] += 1
                        article_sentiments[current_word] += lexicon.loc[sf[m]].at['value']
                        found_words[current_word].append(lexicon.loc[sf[m]].at['words'])

                        break
            else:
                continue
        for word in analysis_words:
            dataset.at[i, f'article_sentiment_for:_{word}'] = article_sentiments[word]
            dataset.at[i, f'n_of_analysis_word_for:_{word}'] = found_analysis_words[word]
            dataset.at[i, f'n_of_lexicon_words_for:_{word}'] = n_of_countable_words[word]
            dataset.at[i, f'lexicon_words_for:_{word}'] = ' '.join(found_words[word])
    return dataset
