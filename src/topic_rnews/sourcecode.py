import pandas as pd


def import_df_from_output(name:str) -> pd.DataFrame:
    return pd.read_csv(f'output/{name}_pre.csv')