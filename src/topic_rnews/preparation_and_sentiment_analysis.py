import read_data
import preprocessing
import sentiment_analysis_fast
# from topic_model_src import gensim_lda, run_leet_topic
from parallel_pandas import ParallelPandas
import datetime
import torch

if __name__ == "__main__":
    # initialize parallel pandas
    ParallelPandas.initialize(n_cpu=256, split_factor=8)
    # import data from numpy arrays
    print(datetime.datetime.now(), 'beginning')
    args = read_data.get_args()
    news_df = read_data.create_dataframe(args)
    print(datetime.datetime.now(), f'dataframe created (size:{len(news_df.index)}), joining headings and paragraphs:')

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # sentiment analysis
    news_df = sentiment_analysis_fast.per_article(news_df)

    words_of_interest = ['estland', 'lettland', 'livland', 'litauen', 'finnland', 'england', 'schweden', 'norwegen', 'daenemark', 'frankreich']
    range_tuple = (5, 25)
    news_df = sentiment_analysis_fast.by_words(news_df, words_of_interest, range_tuple)
    news_df.to_csv(args.output_document_path + '_sent_a.csv', sep=';', index=False)
    # topic modelling
    #news_df = run_leet_topic(news_df, args.leet_distance)
    #print(datetime.datetime.now(), ": finished leet topic model, saving, then starting LDA")
    #news_df.to_csv(args.output_document_path + 'safety_leet_save.csv', sep=';', index=False)
#
 #   news_df = gensim_lda(news_df, args.lda_numtopics)
  #  print(datetime.datetime.now(), ": finished LDA")
#
 #   # export results
  #  news_df.to_csv(args.output_document_path + '.csv', sep=';', index=False)
