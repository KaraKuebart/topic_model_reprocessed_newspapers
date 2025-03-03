import read_data
from topic_model import run_leet_topic
import pandas as pd


def import_df(name:str) -> pd.DataFrame:
    return pd.read_csv(f'output/{name}_pre.csv')


if __name__ == "__main__":
    args = read_data.get_args()

    news_df = import_df(args.batch_nr)
    news_df = run_leet_topic(news_df, args.leet_distance)
    news_df.to_csv(f'output/leet_results{args.batch_nr}.csv', sep=';', index=False)
