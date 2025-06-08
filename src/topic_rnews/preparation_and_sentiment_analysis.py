import read_data
import preprocessing
import sentiment_analysis_src
from parallel_pandas import ParallelPandas
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm
import pickle
import numpy as np


if __name__ == "__main__":
    # initialize parallel pandas
    args = read_data.get_args()
    ParallelPandas.initialize(n_cpu=args.no_of_cpu_cores, split_factor=4)
    # import data from numpy arrays
    print(datetime.datetime.now(), ': beginning import')

    news_df = read_data.create_dataframe(args)
    print(datetime.datetime.now(), f': dataframe created (size:{len(news_df.index)}), joining headings and paragraphs:')

    # add headings and corresponding paragraphs together (if confidence is high).
    news_df = preprocessing.join_headings_w_paragraphs(news_df)

    # drop unuseful data
    news_df = preprocessing.drop_short_lines(news_df)

    # preprocessing: lowercase,
        # consolidations (long ſ -> s, ä ö ü ß -> ae oe ue ss),
        # lemmata (OCR corrections),
        # stopword removal,
        # punctuation removal:
    news_df = preprocessing.main(news_df)
    news_df.reset_index(drop=True, inplace=True)


    # save preliminary results
    news_df.to_csv(args.output_document_path + '_pre.csv', sep=';', index=False)

    # ANALYSIS

    # sentiment analysis
    news_df = sentiment_analysis_src.per_article(news_df)

    words_of_interest = ['estland', 'lettland', 'livland', 'litauen', 'finnland', 'england', 'schweden', 'norwegen', 'daenemark', 'frankreich']
    range_tuple = (5, 25)
    news_df = sentiment_analysis_src.by_words(news_df, words_of_interest, range_tuple)
    news_df.to_csv(args.output_document_path + '_sent_a.csv', sep=';', index=False)

    print(datetime.datetime.now(), ': starting sentiment analysis')
    news_df.sort_values(['path', 'region'], inplace=True)
    news_df.reset_index(drop=True, inplace=True)
    print(datetime.datetime.now(), ': reset index')

    # split into sections
    news_df['first_part'] = news_df['path'].str.split(r'[.-]').str[0]

    # Grouping by the first part
    grouped_dfs = {key: group for key, group in news_df.groupby('first_part')}

    # I decided to keep averages. An alternative would be unweighted sums

    words_of_interest = ['estland', 'lettland', 'livland', 'litauen', 'finnland', 'england', 'schweden', 'norwegen',
                         'daenemark', 'frankreich']

    results = {}
    for word in tqdm(words_of_interest):
        sent_res = {}
        sent_path = f'article_sentiment_for:_{word}'
        #num_sent_res = {}
        num_sent_path = f'n_of_analysis_word_for:_{word}'
        #num_words_res = {}
        #num_words_path = f'n_of_lexicon_words_for:_{word}'
        words_res = {}
        words_path = f'lexicon_words_for:_{word}'
        for year, group_df in grouped_dfs.items():
            length = len(group_df.index)
            sentiment = sum(group_df[sent_path].tolist())
            frequency = sum(group_df[num_sent_path].tolist())
            if frequency > 0:
                sent_res[year] = sentiment / frequency
            else:
                sent_res[year] = 0
            # num_words_res[year] = sum(group_df[num_words_path].tolist()) / length
            wordlist = group_df[words_path].tolist()
            wordlist = list(filter(lambda x: isinstance(x, str), wordlist))
            words_res[year] = ' '.join(wordlist)
        results[sent_path] = sent_res
        results[words_path] = words_res
        title = f'''sentiment results for the word "{word}"'''
        path = f'article_sentiment_for_{word}'
        names = [key for key in sent_res.keys()]
        values = [value for value in sent_res.values()]

        with open('output/sentiment_results.pkl', 'wb') as file:
            pickle.dump(results, file)
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(names, values)

        # Customize the plot
        plt.title(title)
        plt.xlabel('years')
        plt.xticks(np.arange(len(names))[::5], np.array(names)[::5], rotation=45)
        plt.ylabel(title)
        # plt.ylim(0, max(values) * 1.1)  # Set y-axis limit with some padding

        # Add value labels on top of each bar
        #for j, v in enumerate(values):
        #    plt.text(j, v, f'{v:.5f}', ha='center', va='bottom')

        # Display the plot
        plt.tight_layout()
        plt.savefig(f'output/senta_{path}.png')
        plt.close()


        cloud_words = ''
        for year, string in words_res.items():
            if len(string) > 10:
                cloud_words += ' ' + string
        if len(cloud_words) > 0:
            cloud = WordCloud(width=1000, height=600, background_color='white').generate(cloud_words)
            plt.figure(figsize=(10, 6))
            plt.imshow(cloud, interpolation='bilinear')
            plt.savefig(f'output/senta_cloud_{word}.png')
            plt.close()

    # create plot on overall senta res
    sentiment_values = []
    labels = []
    for year, group_df in grouped_dfs.items():
        labels.append(year)
        sentiment_values.append(sum(group_df['sentiment_values'].tolist()) / sum(group_df['n_of_countable_words'].tolist()))
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sentiment_values)
    plt.title('overall sentiment values per year')
    plt.xlabel('years')
    plt.xticks(np.arange(len(labels))[::5], np.array(labels)[::5], rotation=45)
    plt.ylabel('normalized sentiment values')
    #for j, v in enumerate(sentiment_values):
    #    plt.text(j, v, f'{v:.5f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'output/sentiment_values.png')
    plt.close()