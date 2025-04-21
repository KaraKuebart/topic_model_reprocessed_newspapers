import read_data
import preprocessing
from parallel_pandas import ParallelPandas
import datetime


if __name__ == "__main__":
    # initialize parallel pandas
    args = read_data.get_args()
    ParallelPandas.initialize(n_cpu=args.no_of_cpu_cores, split_factor=4)

    # import data from numpy arrays
    news_df = read_data.create_dataframe(args)

    # add headings and corresponding paragraphs together (if confidence is high).
    news_df = preprocessing.join_headings_w_paragraphs(news_df)

    # drop unuseful data (short lines, usually mostly ads and faulty data)
    news_df = preprocessing.drop_short_lines(news_df)

    news_df.to_csv(args.output_document_path + '.csv', sep=';', index=False)
