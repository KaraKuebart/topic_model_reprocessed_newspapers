import read_data
import preprocessing
from parallel_pandas import ParallelPandas
import datetime


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

    news_df.to_csv(args.output_document_path + '.csv', sep=';', index=False)
