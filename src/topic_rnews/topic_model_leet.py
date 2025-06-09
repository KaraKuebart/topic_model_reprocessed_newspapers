from leet_topic import leet_topic
from matplotlib import pyplot as plt

import read_data
from topic_model_src import make_wordcloud
import pandas as pd
from tqdm import tqdm
import datetime


def run_leet_topic(dataframe: pd.DataFrame, max_distance: float=0.5) -> pd.DataFrame:
    new_df, topic_data = leet_topic.LeetTopic(dataframe,
                                               document_field="text",
                                               html_filename=f"output/newspaper_leet_topic_{max_distance}.html",
                                               extra_fields=["hdbscan_labels"],
                                               spacy_model="de_core_news_sm",
                                               max_distance=max_distance)
    return new_df


def count_topic_frequency(topic_names:pd.Series) -> dict:
    return topic_names.value_counts().to_dict()


def topic_scatterplot(df: pd.DataFrame, filename: str) -> None:
    plt.figure(figsize=(25, 15))
    scatter = plt.scatter(df['x'], df['y'], c=df['leet_labels'], cmap='tab10', edgecolors='k',
                          alpha=0.75)
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Topic Number', fontsize=30)
    # Labels and Title
    plt.xlabel('X Coordinate', fontsize=30)
    plt.ylabel('Y Coordinate', fontsize=30)
    plt.title('Topic distribution Visualization', fontsize=30)
    plt.savefig(f'output/{filename}.png')
    plt.close()


def second_leet_topic(df: pd.DataFrame, topic_number) -> pd.DataFrame:
    df_red = df.loc[df['leet_labels'] == topic_number]
    df_red = run_leet_topic(df_red, 0.1)
    df_red.to_csv(f'output/all_double_topic_on_topicnr_{topic_number}.csv', sep=';', index=False)
    print(datetime.datetime.now(), ': finished topic nr.', topic_number)
    return df_red


if __name__ == "__main__":
    args = read_data.get_args()

    dataframe = pd.read_csv(args.load_dataframe, sep=";", on_bad_lines='warn', encoding_errors="ignore")
    dataframe = dataframe.astype(str)
    print(f'{datetime.datetime.now()}: beginning leet topic model')
    dataframe = run_leet_topic(dataframe, args.leet_distance)
    dataframe.to_csv(args.output_document_path + '_leet_res.csv', sep=';', index=False)

    print(f'{datetime.datetime.now()}: creating scatter plots')
    dataframe.sort_values(['path', 'region'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    topic_scatterplot(dataframe, 'leet_topic_scatter')

    topic_dict = count_topic_frequency(dataframe['leet_labels'])

    most_frequent_topics = dict(list(topic_dict.items())[1:101])

    print(f'{datetime.datetime.now()}: creating word clouds for each topic')
    for key in tqdm(most_frequent_topics.keys()):
       try:
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
       except Exception as e:
           print(" an error occurred: ", e)
    most_important_topics_list = list(most_frequent_topics.keys())
    reduced_df = dataframe.loc[dataframe['leet_labels'].isin(most_important_topics_list)]

    topic_scatterplot(reduced_df, 'leet_topic_scatter_reduced')
