import pandas as pd
from loguru import logger
import numpy as np

from typing import List, Dict

def get_dataset_info(df):
    columns = df.columns.tolist()
    types = df.dtypes.apply(lambda x: str(x)).to_dict()
    sample_data = df.head().to_dict(orient='list')
    value_counts = {col: df[col].value_counts().head().to_dict() for col in df.columns}
    description = df.describe().to_dict()

    dataset_info = {
        'columns': columns,
        'types': types,
        'sample_data': sample_data,
        'value_counts': value_counts,
        'description': description
    }
    return dataset_info