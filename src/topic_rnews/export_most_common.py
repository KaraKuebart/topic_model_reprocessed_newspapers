import pandas as pd
from collections import Counter
import pickle
import tqdm
import datetime

from topic_model_src import make_wordcloud

def export_most_common(df: pd.DataFrame, topicnr_column:str):
    topics_list = df[topicnr_column].tolist()
    topics_frequency_dict = dict(Counter(topics_list))
    top_100 = dict(sorted(topics_frequency_dict.items(), key=lambda x: x[1], reverse=True)[:100])
    return top_100


def top_words_by_key(input_dict, top_n=20):
    output_dict = {}
    for key, string_list in input_dict.items():
        # Split each string into words and flatten the list
        words = [word for s in string_list for word in s.split()]
        # Count word frequency and get top N
        most_common = [word for word, _ in Counter(words).most_common(top_n)]
        output_dict[key] = most_common
    return output_dict


if __name__ == '__main__':


    # tomotopy
    print(datetime.datetime.now(), ': importing tomotopy')
    tomoto_df = pd.read_csv('/home/kara/Desktop/köz_results/topic_model_reprocessed_newspapers/analysis_results/tomotopy/tomoto_1140_topics/tomoto_1140_topics.csv', sep=';')
    dict_100 = export_most_common(tomoto_df, '0_topic_nr')
    print(dict_100)
    tomoto_top_100_df = pd.DataFrame(columns=['topic', 'frequency'])
    for key, value in dict_100.items():
        tomoto_top_100_df.loc[len(tomoto_top_100_df)] = [key, value]
    tomoto_top_100_df.set_index('topic', inplace=True)
    del tomoto_df

    tomoto_top_100_df.insert(loc=1, column='representative_words', value='')
    for key in dict_100.keys():
        with open(f'/home/kara/Desktop/köz_results/topic_model_reprocessed_newspapers/analysis_results/tomotopy/tomoto_1140_topics/wordclouds_words/tomoto_words_topic_{int(key)}.pkl', 'rb') as f:
            dict_words = pickle.load(f)
            top_keys = sorted(dict_words, key=dict_words.get, reverse=True)[:20]
        tomoto_top_100_df.at[key, 'representative_words'] = top_keys
    print(tomoto_top_100_df)
    tomoto_top_100_df.to_csv('output/tomoto_top_100.csv', sep=';', index=True)
    del tomoto_top_100_df
    print(datetime.datetime.now(), ': Exported tomotopy. Importing BERTopic')

    # BERTopic
    bert_df = pd.read_csv('/home/kara/Desktop/köz_results/topic_model_reprocessed_newspapers/analysis_results/BERTopic/BERTopic_md00/bert_6th_bertopic_topics.csv', sep=';')
    bert_top_100 = bert_df.head(100)
    del bert_df
    representative_words = []
    for i in bert_top_100['topic_representations']:
        temp_words = []
        i = i[1:-2]
        print(i)
        i = i.split('), (')
        print(i)
        for j in i:
            j = j.replace('(', '').replace(')', '')
            temp_words.append(j.split(',')[0])
        representative_words.append(temp_words)

    bert_top_100.insert(loc=2, column='representative_words', value= representative_words)
    print(bert_top_100)
    bert_top_100.to_csv('output/bertopic_top_100.csv', sep=';', index=True)
    del bert_top_100

    print(datetime.datetime.now(), ': Exported BERTopic. Importing LeetTopic')
    # LeetTopic
    leet_df = pd.read_csv('/home/kara/Desktop/köz_results/topic_model_reprocessed_newspapers/analysis_results/LeetTopic/leet_resultsbatch_1.csv', sep=';', encoding_errors='ignore', on_bad_lines='skip')
    leet_dict_100 = export_most_common(leet_df, 'leet_labels')
    leet_top_100 = pd.DataFrame(columns=['topic', 'frequency'])
    for key, value in leet_dict_100.items():
        leet_top_100.loc[len(leet_top_100)] = [key, value]
    leet_top_100.set_index('topic', inplace=True)
    topics_100 = leet_dict_100.keys()
    words_dict_100 = {key: [] for key in topics_100}
    topics_100 = [int(x) for x in topics_100]
    for i in leet_df.index:
        if leet_df['leet_labels'][i] in topics_100:
            words_dict_100[float(leet_df['leet_labels'][i])].append(leet_df['text'][i])
    del leet_df

    final_dict = top_words_by_key(words_dict_100, top_n=20)
    leet_top_100.insert(loc=1, column='representative_words', value= '')
    for key in final_dict.keys():
        leet_top_100['representative_words'][key] = final_dict[key]
    print(leet_top_100)
    leet_top_100.to_csv('output/leettopic_top_100.csv', sep=';', index=True)
