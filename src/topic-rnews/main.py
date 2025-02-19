import pandas as pd
from tqdm import tqdm


import read_data
import preprocessing

if __name__ == "__main__":
    # import data from numpy arrays
    news_df = read_data.create_dataframe(read_data.get_args())

    # Sorting the df by date and reading order
    news_df.sort_values(by=['path', 'region'], inplace=True)

    # add headings and corresponding paragraphs together (if confidence is high).
    for i in news_df.index:
        if news_df.loc[i]['class'] == 'heading' and news_df.loc[i+1]['class'] == 'paragraph' and news_df.loc[i]['confidence'] > 0.5 and news_df.loc[i+1]['confidence'] > 0.5:
            news_df.loc[i]['class'] = 'joined'
            news_df.loc[i]['confidence'] = (news_df.loc[i]['confidence'] + news_df.loc[i+1]['confidence']) / 2.0
            news_df.loc[i]['text'] = news_df.loc[i]['text'] + ' ' + news_df.loc[i+1]['text']




    # preprocessing: lowercase,
        # consolidations (long ſ -> s, äöüß -> aeoeuess),
        # lemmata (OCR corrections),
        # stopword removal,
        # punctuation removal:
    news_df = preprocessing.main(news_df)


    # drop unuseful data
    print('before_reduction:', len(news_df))

    for i in tqdm(news_df.index):
        if type(news_df.loc[i, "text"]) != str or len(news_df.loc[i, "text"]) < 50:
            news_df.drop(i, inplace=True)
    print('after reduction:', len(news_df))


    # export results
    news_df.to_csv('temp.csv', sep=';')