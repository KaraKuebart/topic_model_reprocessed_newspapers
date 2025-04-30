import read_data
from topic_model_src import run_leet_topic, topic_scatterplot, count_topic_frequency, make_wordcloud
import pandas as pd
from tqdm import tqdm
import datetime

if __name__ == "__main__":
    args = read_data.get_args()

    dataframe = pd.read_csv(args.load_dataframe, sep=";", on_bad_lines='warn', encoding_errors="ignore")

    print(f'{datetime.datetime.now()}: beginning leet topic model')
    dataframe = run_leet_topic(dataframe, args.leet_distance)
    dataframe.to_csv(args.output_document_path + '_leet_res.csv', sep=';', index=False)

    print(f'{datetime.datetime.now()}: creating scatter plots')
    dataframe.sort_values(['path', 'region'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    topic_scatterplot(dataframe, 'leet_topic_scatter')

    topic_dict = count_topic_frequency(dataframe['leet_labels'])

    most_frequent_topics = dict(list(topic_dict.items())[1:100])

    print(f'{datetime.datetime.now()}: creating word clouds for each topic')
    for key in tqdm(most_frequent_topics.keys()):
        temp_df = dataframe.loc[dataframe['leet_labels'] == key]
        temp_df.to_csv(f"output/leet_topic_{key}.csv", sep=';', index=False)
        textlist = temp_df['text'].tolist()
        texts_string = '\n'.join(textlist)
        texts_string.replace('\n', ' ')
        wordlist = texts_string.split(' ')
        word_frequencies = {}
        word_set = set(wordlist)
        for item in word_set:
            word_frequencies[item] = wordlist.count(item)

        make_wordcloud('leet', key, word_frequencies)

    most_important_topics_list = list(most_frequent_topics.keys())
    reduced_df = dataframe.loc[dataframe['leet_labels'].isin(most_important_topics_list)]

    topic_scatterplot(reduced_df, 'leet_topic_scatter_reduced')
