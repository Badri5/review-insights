"""Generate a synthetic review file for testing (same columns as the Kaggle dataset).

    python data/make_sample.py            -> data/sample_reviews.csv (5,000 rows)
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

random.seed(7)

POSITIVE = [
    "Great taste and the quality is excellent.",
    "Love the flavour, my kids ask for it every day.",
    "Fair price for what you get, will buy again.",
    "Packaging was neat and delivery was quick.",
    "Healthy snack that actually tastes good.",
    "Fresh and crunchy, highly recommend.",
    "Best coffee I have had in years.",
]
NEGATIVE = [
    "Terrible taste, it was stale when it arrived.",
    "Packaging was broken and half the box was empty.",
    "Way too expensive for such poor quality.",
    "Delivery took three weeks, never ordering again.",
    "Disappointing, nothing like the description.",
    "Awful aftertaste, a complete waste of money.",
    "The worst batch I have ever received.",
]
FILLER = ["", " Ordered it last month.", " Bought two packs.", " My second order.", " Gift for a friend."]
PRODUCTS = ["green tea", "oat cookies", "dark chocolate", "dog treats", "almond butter", "instant noodles",
            "coffee beans", "protein bars", "honey", "trail mix", "olive oil", "rice crackers"]


def review_text(positive: bool) -> str:
    """Two sentences plus product and context, so texts are varied like real reviews."""
    pool = POSITIVE if positive else NEGATIVE
    a, b = random.sample(pool, 2)
    product = random.choice(PRODUCTS)
    return f"Bought the {product}. {a} {b}{random.choice(FILLER)} ({random.randint(1, 99)} days in.)"


def make(n=5000, months=48):
    start = datetime(2022, 1, 1)
    rows = []
    for i in range(n):
        # Volume grows over time with a year-end bump, so the forecast has a trend to learn.
        month = min(int(random.triangular(0, months, months)), months - 1)
        day = start + timedelta(days=30 * month + random.randint(0, 27))
        if day.month in (11, 12) and random.random() < 0.25:
            day = day.replace(day=min(day.day, 28))
        positive = random.random() < 0.78
        # ~8% label noise so models can't score a perfect 1.0
        flip = random.random() < 0.08
        text = review_text(positive != flip)
        score = random.choice([4, 5] if positive else [1, 2])
        if random.random() < 0.1:
            score = 3
        rows.append({
            "Id": i + 1,
            "Score": score,
            "Time": int(day.timestamp()),
            "Summary": text.split(",")[0],
            "Text": text,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_reviews.csv"
    make().to_csv(out, index=False)
    print(f"wrote {out}")
