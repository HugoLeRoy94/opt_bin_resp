# %%
"""
Load a runs.db into a pandas DataFrame.

Usage
-----
    # as a script
    python analysis/load_db.py /path/to/runs.db

    # in a notebook
    from analysis.load_db import load_db
    df = load_db("/path/to/runs.db")
"""
import sqlite3
import sys
import pandas as pd


def load_db(db_path: str, where: str = "status='complete'") -> pd.DataFrame:
    """Return a DataFrame with one row per run in runs.db.

    Metric columns are renamed from '{metric}_mean' to '{metric}' to match
    the convention used by SweepLoader.load_all_test_results().

    Parameters
    ----------
    db_path : path to runs.db
    where   : SQL WHERE clause to filter rows (default: complete runs only).
              Pass "" or None to return all rows.
    """
    sql = "SELECT * FROM runs"
    if where:
        sql += f" WHERE {where}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    df = df.rename(columns={c: c[:-5] for c in df.columns if c.endswith("_mean")})
    return df
#%%

with sqlite3.connect('/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/data/test/runs.db') as conn:
        df = pd.read_sql_query("SELECT * FROM runs WHERE status='complete'", conn)

# %%
print(df['status'])
# %%
