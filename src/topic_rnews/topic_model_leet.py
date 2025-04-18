import read_data
from topic_model_src import run_leet_topic, topic_scatterplot, count_topic_frequency
import pandas as pd
from wordcloud import WordCloud
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = read_data.get_args()

    dataframe = pd.read_csv(args.load_dataframe, sep=";", on_bad_lines='warn', encoding_errors="ignore")
    dataframe = run_leet_topic(dataframe, args.leet_distance)
    filename = args.load_dataframe.split('.')[0]
    dataframe.to_csv(f'{filename}_leet_res.csv', sep=';', index=False)


    dataframe.sort_values(['path', 'region'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    topic_scatterplot(dataframe, 'leet_topic_scatter')

    topic_dict = count_topic_frequency(dataframe['leet_labels'])
    print(topic_dict)
    most_important_topics = dict(list(topic_dict.items())[1:100])
    print(most_important_topics)
    for key in most_important_topics.keys():
        temp_df = dataframe.loc[dataframe['leet_labels'] == key]
        temp_df.to_csv(f"output/leet_topic_{key}.csv", sep=';', index=False)
        textlist = temp_df['text'].tolist()
        textstring = '\n'.join(textlist)
        textstring.replace('\n', ' ')
        wordlist = textstring.split(' ')
        word_frequencies = {}
        word_set = set(wordlist)
        for item in word_set:
            word_frequencies[item] = wordlist.count(item)

        with open(f'output/leet_words_topic_{key}.pkl', 'wb') as file:
            pickle.dump(word_frequencies, file)
        cloud = WordCloud(width=1000, height=600, background_color='white').generate(textstring)
        plt.figure(figsize=(10, 6))
        plt.imshow(cloud, interpolation='bilinear')
        plt.title(f'Words in Topic {key}', fontsize=30)
        plt.savefig(f'output/leet_words_topic_{key}.png')
        plt.close()

    most_important_topics_list = list(most_important_topics.keys())
    reduced_df = dataframe.loc[dataframe['leet_labels'].isin(most_important_topics_list)]

    topic_scatterplot(reduced_df, 'leet_topic_scatter_reduced')
