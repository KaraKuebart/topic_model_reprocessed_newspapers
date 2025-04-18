import pandas as pd
from read_data import get_args
from topic_model_src import tomoto_lda

if __name__ == "__main__":
    # get args for import
    args = get_args()

    # import dataframe
    news_df = pd.read_csv(args.load_dataframe, sep=';')
    # run tomoto LDAModel
    news_df = tomoto_lda(news_df)
    # save results file
    filename = args.load_dataframe.split('.')[0]
    news_df.to_csv(filename + '_tomoto_res.csv', index=False)
