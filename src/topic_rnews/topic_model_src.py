import pickle
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import tomotopy as tp
from leet_topic import leet_topic
from wordcloud import WordCloud


def run_leet_topic(dataframe: pd.DataFrame, max_distance: float=0.5) -> pd.DataFrame:
    new_df, topic_data = leet_topic.LeetTopic(dataframe,
                                               document_field="text",
                                               html_filename=f"output/newspaper_leet_topic_{max_distance}.html",
                                               extra_fields=["hdbscan_labels"],
                                               spacy_model="de_core_news_sm",
                                               max_distance=max_distance)
    return new_df


def second_leet_topic(df: pd.DataFrame, topic_number) -> pd.DataFrame:
    df_red = df.loc[df['leet_labels'] == topic_number]
    df_red = run_leet_topic(df_red, 0.1)
    df_red.to_csv(f'output/all_double_topic_on_topicnr_{topic_number}.csv', sep=';', index=False)
    print(datetime.datetime.now(), ': finished topic nr.', topic_number)
    return df_red


def count_topic_frequency(topic_names:pd.Series) -> dict:
    return topic_names.value_counts().to_dict()


def topic_scatterplot(df: pd.DataFrame, filename: str) -> None:
    global scatter, cbar
    plt.figure(figsize=(25, 15))
    scatter = plt.scatter(df['x'], df['y'], c=df['leet_labels'], cmap='tab10', edgecolors='k',
                          alpha=0.75)
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Topic Number', fontsize=30)
    # Labels and Title
    plt.xlabel('X Coordinate', fontsize=30)
    plt.ylabel('Y Coordinate', fontsize=30)
    plt.title('Topic Visualization', fontsize=30)
    plt.savefig(f'output/{filename}.png')
    plt.close()




def tomoto_lda(dataframe: pd.DataFrame, num_topics:int=None, out_filename:str='tomotopy_lda'):
    num_docs = dataframe.shape[0]
    if num_topics is None:
        num_topics = round(np.power(num_docs, 5/12))
    mdl = tp.LDAModel(min_cf = int(np.power(num_docs, 1/3)), rm_top=int(np.log(num_docs)*20), k=num_topics, seed=42)
    print('importing data into tomoto_lda model')
    for i in tqdm(dataframe.index):
        text = str(dataframe.at[i, 'text']).split()
        mdl.add_doc(text)
    for j in range(0, 100, 10):
        mdl.train(10)
    print('removed top words', mdl.removed_top_words, '\n', '')
    mdl.save('output/' + out_filename +'.bin')
    for k in tqdm(dataframe.index):
        doc_inst= mdl.docs[k]
        print(doc_inst)
        print(str(dataframe.at[k, 'text']).split())

        topic_dists = doc_inst.get_topic_dist()
        topic_tuplist = []
        for l, prob in enumerate(topic_dists):
            topic_tuplist.append((l, round(prob, 4)))
        topic_tuplist.sort(reverse=True, key=lambda tup: tup[1])
        for m in range(0, 4):
            dataframe.at[k, f'{m}_topic_nr'] = topic_tuplist[m][0]
            dataframe.at[k, f'{m}_topic_probability'] = topic_tuplist[m][1]
    return dataframe, mdl, num_topics

# TODO: implement inference (to train on a fraction of the data, then infer to the rest


def make_wordcloud(method:str, topic_number:int, frequencies:dict)-> None:
    with open(f'output/leet_words_topic_{topic_number}.pkl', 'wb') as file:
        pickle.dump(frequencies, file)
    cloud = WordCloud(width=1000, height=600, background_color='white').generate_from_frequencies(frequencies)
    plt.figure(figsize=(10, 6))
    plt.imshow(cloud, interpolation='bilinear')
    plt.title(f'Words in Topic {topic_number}', fontsize=30)
    plt.savefig(f'output/{method}_words_topic_{topic_number}.png')
    plt.close()
