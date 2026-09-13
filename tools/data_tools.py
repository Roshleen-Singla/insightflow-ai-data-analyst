import pandas as pd
from langchain_core.tools import tool


@tool
def get_dataset_info(file_path: str) -> str:
    """
    Provides a readable summary of the dataset.
    """

    df = pd.read_csv(file_path, low_memory=False)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    info = f"""
DATASET SUMMARY
---------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

COLUMNS
-------
{", ".join(df.columns)}

DATA TYPES
----------
{df.dtypes.to_string()}

MISSING VALUES
--------------
{missing.to_string() if not missing.empty else "No missing values"}

INFORMATION ABOUT NUMERIC COLUMNS
-------------------------------
{df.describe().to_string() if not df.select_dtypes(include='number').empty else "No numeric columns"}

DUPLICATE ROWS
----------------
{df.duplicated().sum()}
"""

    return info