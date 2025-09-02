import datetime

import pandas as pd
import psutil

import read_data
from preprocessing import drop_short_lines
from topic_model_leet import run_leet_topic, leet_post_processing

if __name__ == "__main__":
    args = read_data.get_args()

    print(datetime.datetime.now(), f': importing dataframe. RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')
    dataframe = pd.read_csv(args.load_dataframe, sep=";", on_bad_lines='warn', encoding_errors="replace")
    dataframe = dataframe.astype(str)

    print(datetime.datetime.now(), f': raw dataframe loaded. dropping short lines. RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')
    dataframe = drop_short_lines(dataframe)
    dataframe.reset_index(drop=True, inplace=True)

    print(f'{datetime.datetime.now()}: beginning leet topic . RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')
    dataframe = run_leet_topic(dataframe, args.leet_distance)

    leet_post_processing(dataframe, args)
