import pandas as pd
from read_data import get_args
from topic_model_src import tomoto_lda, make_wordcloud

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
    news_df.to_csv(args.output_document_path + '_tomoto_res.csv', index=False)
    vocabulary = mdl.vocabs
    words_importance_dict = {}
    print(f'{datetime.datetime.now()}: creating word clouds for each topic')
    for topic_id in tqdm(range(num_topics)):
        topic_words = mdl.get_topic_word_dist(topic_id, normalize=True)
        for w in range(len(topic_words)):
            words_importance_dict[vocabulary[w]] = topic_words[w]

        make_wordcloud('tomoto', topic_id, words_importance_dict)
