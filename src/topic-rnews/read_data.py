import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm


def merge_data(data_path: Path, paths: List[str]) -> np.ndarray:
    page_list = []
    for path in tqdm(paths[:200]):
        page_csv = np.load(data_path / f"{path}.npz")['array']
        page_list.append(page_csv)
    return np.vstack(page_list)


def create_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load xml files and assemble page lists before saving them.
    """
    data_path = Path(args.data_path)

    paths = [
        f[:-4] for f in os.listdir(data_path) if f.endswith(".npz")
    ]

    data = merge_data(data_path, paths)
    df = pd.DataFrame(data, columns=["path", "region", "class", "confidence", "text"])
    df.to_csv(args.original_dataframe, index=False)
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
        default="original_dataframe.csv",
        help="saving path for the original dataframe without edits",
    )
    return parser.parse_args()
