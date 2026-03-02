"""
preprocessing.py
Functions to load and preprocess raw neuroscience data.
"""

import numpy as np
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    \"\"\"Load data from CSV/TSV or custom format. Return a pandas DataFrame.\"\"\"
    # Replace with real loader (hdf5, nibabel, mat, etc) as needed
    return pd.read_csv(path)

def basic_clean(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    \"\"\"Simple cleaning step: drop duplicates, optionally drop NA.\"\"\"
    df = df.drop_duplicates().reset_index(drop=True)
    if dropna:
        df = df.dropna()
    return df

def set_random_seeds(seed: int = 42):
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
