import pandas as pd
from read_data import get_args
from topic_model_src import tomoto_lda
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

if __name__ == "__main__":
    # get args for import
    args = get_args()

    # import dataframe
    news_df = pd.read_csv(args.load_dataframe, sep=';')
    # run tomoto LDAModel
    news_df, mdl, num_topics = tomoto_lda(news_df)
    # save results file
    filename = args.load_dataframe.split('.')[0]
    news_df.to_csv(filename + '_tomoto_res.csv', index=False)
    vocabulary = mdl.vocabs
    words_importance_dict = {}
    print(f'{datetime.datetime.now()}: creating wordclouds for each topic')
    for topic_id in tqdm(range(num_topics)):
        topic_words = mdl.get_topic_word_dist(topic_id, normalize=True)
        for w in range(len(topic_words)):
            words_importance_dict[vocabulary[w]] = topic_words[w]
        with open(f'output/tomoto_words_topic_{topic_id}.pkl', 'wb') as file:
            pickle.dump(words_importance_dict, file)


        cloud = WordCloud(width=1000, height=600, background_color='white').generate_from_frequencies(words_importance_dict)

        plt.figure(figsize=(10, 6))
        plt.imshow(cloud, interpolation='bilinear')
        plt.title(f'Words characteristic for topic Nr. {topic_id}', fontsize=30)
        plt.savefig(f'output/tomoto_words_topic_{topic_id}.png')
        plt.close()
