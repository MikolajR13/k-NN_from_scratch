import numpy as np
import pandas as pd


def normalize_dataframe(df):
    # data from DataFrame min-max scaling
    X_arr = df.values
    min_val = np.min(X_arr, axis=0)
    max_val = np.max(X_arr, axis=0)
    X_normalized = (X_arr - min_val) / (max_val - min_val)
    data_normalized = pd.DataFrame(X_normalized, columns=df.columns, index=df.index)

    return data_normalized
