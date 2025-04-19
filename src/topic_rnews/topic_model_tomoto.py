import pandas as pd
from read_data import get_args
from topic_model_src import tomoto_lda

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
    # TODO implement visualization of results directly from model. Get tomoto_lda to put out model for the purpose.
    words_used_vocab = mdl.used_vocab_df
    vocabulary = mdl.vocabs
    words_importance_dict = {}
    for topic_id in range(num_topics):
        topic_words = mdl.get_topic_word_dist(topic_id, normalize=True)
        for w in range(len(topic_words)):
            # this will turn it into a dict with all needed info. However - we want a wordlist for wordcloud. or can it take a dict?
            words_importance_dict[vocabulary[w]] = topic_words[w]

        print('breakpoint', topic_id)