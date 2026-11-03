"""Load the PaySim dataset and split it by time.

PaySim simulates a month of mobile-money transactions. Each ``step`` is one
hour of simulated time, which gives us a clock to split on: the model must
only ever learn from the past and be judged on the future.

Run ``python -m app.data`` once to convert the Kaggle CSV to Parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings

TARGET = "is_fraud"
TIME_COL = "step"

# Kaggle column name -> our snake_case name
COLUMN_NAMES: dict[str, str] = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

# Explicit types keep memory down on 6M+ rows and catch malformed files early
RAW_DTYPES: dict[str, str] = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

KAGGLE_URL = "https://www.kaggle.com/datasets/ealaxi/paysim1"


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read the PaySim CSV with fixed types and snake_case column names."""
    header = pd.read_csv(path, nrows=0).columns
    missing = set(COLUMN_NAMES) - set(header)
    if missing:
        raise ValueError(f"{path.name} is missing PaySim columns: {sorted(missing)}")
    df = pd.read_csv(path, usecols=list(COLUMN_NAMES), dtype=RAW_DTYPES)
    renamed: pd.DataFrame = df.rename(columns=COLUMN_NAMES)[list(COLUMN_NAMES.values())]
    return renamed


def prepare(raw_dir: Path | None = None, out_path: Path | None = None) -> Path:
    """Convert the single CSV in ``raw_dir`` to Parquet and return its path."""
    raw_dir = raw_dir or settings.raw_dir
    out_path = out_path or settings.processed_path

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found in {raw_dir}. Download PaySim from {KAGGLE_URL} and unzip it there."
        )
    if len(csvs) > 1:
        raise ValueError(f"Expected one CSV in {raw_dir}, found {len(csvs)}: {csvs}")

    df = read_raw_csv(csvs[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def load_paysim(path: Path | None = None) -> pd.DataFrame:
    """Load the prepared Parquet file."""
    path = path or settings.processed_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m app.data` first.")
    return pd.read_parquet(path)


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def summary(self, time_col: str = TIME_COL, target: str = TARGET) -> pd.DataFrame:
        """One row per split: size, time range and fraud rate."""
        rows = [
            {
                "split": name,
                "rows": len(part),
                "first_step": int(part[time_col].min()),
                "last_step": int(part[time_col].max()),
                "fraud_rate": float(part[target].mean()),
            }
            for name, part in (("train", self.train), ("valid", self.valid), ("test", self.test))
        ]
        table: pd.DataFrame = pd.DataFrame(rows).set_index("split")
        return table


def time_split(
    df: pd.DataFrame,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    time_col: str = TIME_COL,
) -> TimeSplit:
    """Split rows into train, validation and test sets in time order.

    Every validation row is later than every training row, and every test row
    is later than every validation row. Rows from the same time step always
    land in the same set, so the fractions are approximate.
    """
    if not (0 < valid_frac < 1 and 0 < test_frac < 1 and valid_frac + test_frac < 1):
        raise ValueError("valid_frac and test_frac must be in (0, 1) and sum to less than 1")

    # Share of all rows at or before each step, in time order
    counts = df[time_col].value_counts().sort_index()
    cum_share = (counts.cumsum() / counts.sum()).to_numpy()
    steps = counts.index.to_numpy()

    train_end = steps[np.searchsorted(cum_share, 1 - valid_frac - test_frac)]
    valid_end = steps[np.searchsorted(cum_share, 1 - test_frac)]

    train = df[df[time_col] <= train_end]
    valid = df[(df[time_col] > train_end) & (df[time_col] <= valid_end)]
    test = df[df[time_col] > valid_end]

    if train.empty or valid.empty or test.empty:
        raise ValueError(
            f"Not enough distinct values of {time_col!r} to make three non-empty splits"
        )
    return TimeSplit(train=train, valid=valid, test=test)


if __name__ == "__main__":
    out = prepare()
    print(f"Wrote {out}")
