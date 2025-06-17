import read_data
import pandas as pd
import datetime

from preprocessing import drop_short_lines
from topic_model_leet import run_leet_topic, count_topic_frequency, topic_scatterplot, leet_post_processing

if __name__ == "__main__":
    args = read_data.get_args()

    print(datetime.datetime.now(), ': importing dataframe')
    dataframe = pd.read_csv(args.load_dataframe, sep=";", on_bad_lines='warn', encoding_errors="ignore")
    dataframe = dataframe.astype(str)

    print(datetime.datetime.now(), ': raw dataframe loaded. dropping short lines')
    dataframe = drop_short_lines(dataframe)

    print(f'{datetime.datetime.now()}: beginning leet topic model')
    dataframe = run_leet_topic(dataframe, args.leet_distance)

    leet_post_processing(dataframe, args)
