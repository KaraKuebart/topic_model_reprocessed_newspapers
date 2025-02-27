import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm


def merge_data(data_path: Path, paths: List[str], import_range: list) -> list:
    page_list = []
    for path in tqdm(paths[import_range[0]:import_range[1]]):
        page_csv = np.load(data_path / f"{path}.npz")['array'].tolist()
        page_list.extend(page_csv)
    return page_list


def create_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load xml files and assemble page lists before saving them.
    """
    data_path = Path(args.data_path)

    paths = [
        f[:-4] for f in os.listdir(data_path) if f.endswith(".npz")
    ]
    import_tuple = args.import_range
    data = merge_data(data_path, paths, import_tuple)
    df = pd.DataFrame(data, columns=["path", "region", "class", "confidence", "text"])
    df.to_csv(args.original_dataframe, sep= ';', index=False)
    return df


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/",
        help="path for data folder",
    )
    parser.add_argument(
        "--original-dataframe",
        "-od",
        type=str,
        default="output/original_dataframe.csv",
        help="saving path for the original dataframe without edits",
    )
    parser.add_argument(
        "--lda-numtopics",
        "-lda",
        type=int,
        default=50,
        help="number of topics for LDA",
    )
    parser.add_argument(
        "--leet-distance",
        "-leet",
        type=float,
        default=0.5,
        help="maximum allowed distance between document and topic for leet topic model",
    )
    parser.add_argument(
        "--import-range",
        "-imp",
        nargs="+",
        type=int,
        default=[0, -1],
        help="defines which part of the files contained in the data folder will be imported",
    )
    parser.add_argument(
        "--output-document-path",
        "-out",
        type=str,
        default="output/out_df",
        help="path to the output dataframe. A temporary file will be created under the same path witch '_pre.csv' added.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    get_args()