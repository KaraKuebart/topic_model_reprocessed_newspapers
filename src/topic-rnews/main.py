import pandas as pd
from tqdm import tqdm


import read_data
import preprocessing
from topic_model import run_lda, run_leet_topic


if __name__ == "__main__":
    # import data from numpy arrays
    args = read_data.get_args()

    news_df = read_data.create_dataframe(args)

    # Sorting the df by date and reading order (has no effect - index not changed)
    # news_df.sort_values(by=['path', 'region'], inplace=True)

    # add headings and corresponding paragraphs together (if confidence is high).
    for i in news_df.index:
        if news_df.loc[i]['class'] == 'heading' and news_df.loc[i+1]['class'] == 'paragraph' and float(news_df.loc[i]['confidence']) > 0.5 and float(news_df.loc[i+1]['confidence']) > 0.5 and news_df.loc[i]['region'] == int(news_df.loc[i+1]['region']) - 1:
            news_df.loc[i]['class'] = 'joined'
            news_df.loc[i]['confidence'] = (news_df.loc[i]['confidence'] + news_df.loc[i+1]['confidence']) / 2.0
            news_df.loc[i]['text'] = str(news_df.loc[i]['text']) + ' ' + str(news_df.loc[i+1]['text'])
            news_df.drop(news_df.loc[i+1], inplace=True)

    # drop unuseful data
    print('before_reduction:', len(news_df))

    for i in tqdm(news_df.index):
        if type(news_df.loc[i, "text"]) != str or len(news_df.loc[i, "text"]) < 50:
            news_df.drop(i, inplace=True)
    print('after reduction:', len(news_df))

    # preprocessing: lowercase,
        # consolidations (long ſ -> s, ä ö ü ß -> ae oe ue ss),
        # lemmata (OCR corrections),
        # stopword removal,
        # punctuation removal:
    news_df = preprocessing.main(news_df)


    # analysis
    news_df = run_lda(news_df, args.lda_numtopics)
    news_df = run_leet_topic(news_df, args.leet_distance)


    # export results
    news_df.to_csv('temp.csv', sep=';')
    print('done')