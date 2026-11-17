"""Load raw review data and turn it into a clean, consistent table with Pandas.

Works with the Kaggle "Amazon Fine Food Reviews" file (Reviews.csv) and with
the sample file made by data/make_sample.py. Output columns:
    review_id, date, rating, review_text, label  (label: 1 = positive, 0 = negative)
"""
import re

import pandas as pd

# Map known source column names to our standard names.
COLUMN_MAP = {
    "Id": "review_id",
    "Score": "rating",
    "Text": "review_text",
    "Time": "date",
}


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))  # strip HTML tags like <br />
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_clean(csv_path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=max_rows)
    df = df.rename(columns=COLUMN_MAP)

    missing = {"rating", "review_text", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {sorted(missing)}")

    if "review_id" not in df.columns:
        df["review_id"] = range(1, len(df) + 1)

    # Kaggle stores time as a Unix timestamp; the sample file uses ISO dates.
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="s")
    else:
        df["date"] = pd.to_datetime(df["date"])

    df["review_text"] = df["review_text"].map(_clean_text)
    df = df.dropna(subset=["rating", "review_text"])
    df = df[df["review_text"].str.len() > 10]
    df = df.drop_duplicates(subset=["review_text"])

    # 4-5 stars = positive, 1-2 = negative; 3-star reviews are ambiguous, so drop them.
    df["rating"] = df["rating"].astype(int)
    df = df[df["rating"] != 3].copy()
    df["label"] = (df["rating"] >= 4).astype(int)

    cols = ["review_id", "date", "rating", "review_text", "label"]
    return df[cols].sort_values("date").reset_index(drop=True)
