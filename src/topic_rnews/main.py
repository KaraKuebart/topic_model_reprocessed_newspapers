import read_data
import preprocessing
import sentiment_analysis
from topic_model import run_lda, run_leet_topic
from tqdm import tqdm

if __name__ == "__main__":
    # import data from numpy arrays
    args = read_data.get_args()
    news_df = read_data.create_dataframe(args)

    # add headings and corresponding paragraphs together (if confidence is high).
    for i in tqdm(news_df.index):
        if news_df.loc[i]['class'] == 'heading' and news_df.loc[i+1]['class'] == 'paragraph' and float(news_df.loc[i]['confidence']) > 0.5 and float(news_df.loc[i+1]['confidence']) > 0.5 and news_df.loc[i]['region'] == int(news_df.loc[i+1]['region']) - 1:
            news_df.loc[i]['class'] = 'joined'
            news_df.loc[i]['confidence'] = (news_df.loc[i]['confidence'] + news_df.loc[i+1]['confidence']) / 2.0
            news_df.loc[i]['text'] = str(news_df.loc[i]['text']) + ' ' + str(news_df.loc[i+1]['text'])
            news_df.drop(news_df.loc[i+1], inplace=True)

    # drop unuseful data
    news_df = preprocessing.drop_short_lines(news_df)

    # preprocessing: lowercase,
        # consolidations (long ſ -> s, ä ö ü ß -> ae oe ue ss),
        # lemmata (OCR corrections),
        # stopword removal,
        # punctuation removal:
    news_df = preprocessing.main(news_df)
    news_df.reset_index(drop=True, inplace=True)


    # ANALYSIS

    # sentiment analysis
    news_df = sentiment_analysis.per_article(news_df)

    words_of_interest = ['estland', 'lettland', 'livland', 'litauen', 'finnland', 'england', 'schweden', 'norwegen', 'daenemark', 'frankreich']
    range_tuple = (0, 30)
    news_df = sentiment_analysis.by_words(news_df, words_of_interest, range_tuple)

    # topic modelling
    news_df = run_leet_topic(news_df, args.leet_distance)
    print("finished leet topic model, saving, then starting LDA")

    news_df = run_lda(news_df, args.lda_numtopics)
    print("finished LDA")
    news_df.to_csv('output/safety_leet_save.csv', sep=';', index=False)
    # export results
    news_df.to_csv('output/koelnische_zeitung_all.csv', sep=';', index=False)
