"""this code serves to add text to files provided via zenodo which contain metadata and topics, but no document text. The text can be read from the raw file named "all_raw.csv"
and added to the other .csv files.
Simply put the raw file into the input directory and all files you wish to add text to in
 the output directory and set the path to this repository as working directory, then run this script.
Be aware you need at least 32 GB of ram, if you integrate text into large files, like the complete dataset, you may need more.

This also works with the preprocessed text, just replace the path in the raw_df import to that of the preprocessed data."""
import pandas as pd
from tqdm import tqdm
import glob
from parallel_pandas import ParallelPandas
import datetime

ParallelPandas.initialize(n_cpu=96, split_factor=4, disable_pr_bar=False)

def insert_row_text(row):
    return raw_df.at[row['ID'], 'text']


def clean_df(df):
    if 'text' in df.columns:
        df.drop(columns=['text'], inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0.1' in df.columns:
        df.drop(columns=['Unnamed: 0.1'], inplace=True)
    if 'Unnamed: 0.01' in df.columns:
        df.drop(columns=['Unnamed: 0.01'], inplace=True)
    df['ID'] = df['path'].astype(str) + '_' + df['region'].astype(str)
    return df

if __name__ == "__main__":
    # read all_raw.csv
    raw_df = pd.read_csv('input/all_raw.csv', sep=';')
    raw_df['ID'] = raw_df['path'].astype(str) + '_' + raw_df['region'].astype(str)
    raw_df.set_index('ID', inplace=True)
    csvfiles = glob.glob('output/*.csv')
    for csvfile in tqdm(csvfiles):
        print(datetime.datetime.now(), f': reintegrating text into: {csvfile}')
        df = pd.read_csv(csvfile, sep=';')
        df = clean_df(df)
        df['text'] = df.p_apply(insert_row_text, axis=1)
        df.to_csv(csvfile, sep=';', index=False)
