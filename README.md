# Customer Review Insights Pipeline

An end-to-end data pipeline that turns raw customer reviews into business insights, combining
**AWS S3/Athena**, **Generative AI (OpenAI vs Google Gemini)**, **classical machine learning**
and **time series forecasting**.

```
 Raw CSV ──► Amazon S3 ──► Pandas cleaning ──┬─► OpenAI + Gemini: topic, sentiment, summary (compared)
                                             ├─► Logistic Regression / Decision Tree / XGBoost (compared)
                                             └─► ARIMA forecast of monthly review volume
                                                          │
                       Amazon Athena (SQL) ◄── S3 ◄───────┴──► Plotly dashboard (HTML)
```

## What it does

| Step | Tools | Output |
| --- | --- | --- |
| 1. Ingest | boto3, Amazon S3 | Raw file stored in `s3://<bucket>/raw/` |
| 2. Clean | Pandas | Deduplicated reviews, HTML stripped, labelled positive/negative |
| 3. GenAI enrichment | OpenAI API, Google Gemini API | Topic, sentiment and summary per review; accuracy, speed and token use compared |
| 4. Machine learning | scikit-learn, XGBoost, TF-IDF | Logistic Regression vs Decision Tree vs XGBoost on accuracy, F1, ROC-AUC |
| 5. Time series | statsmodels ARIMA | 6-month forecast of review volume with 80% interval, back-tested MAE |
| 6. Report | Plotly, Amazon Athena | `outputs/dashboard.html`; results in `s3://<bucket>/processed/` queryable by SQL |

## Quick start (no AWS or API keys needed)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/make_sample.py
python run_pipeline.py --input data/sample_reviews.csv --local --mock-llm
```

Open `outputs/dashboard.html` in a browser.

## Full run with AWS, OpenAI and Gemini

1. **Data:** download `Reviews.csv` from Kaggle's
   [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) into `data/`.
2. **AWS:** create an S3 bucket (free tier is enough) and run `aws configure` with an IAM user
   that can read/write that bucket.
3. **Keys:** copy `.env.example` to `.env` and fill in `S3_BUCKET`, `OPENAI_API_KEY` and `GEMINI_API_KEY`.
4. **Run:**
   ```bash
   python run_pipeline.py --input data/Reviews.csv --max-rows 50000 --llm-sample 100
   ```
5. **Athena:** open `sql/athena_queries.sql`, replace `YOUR_BUCKET`, and run the statements in the
   Athena console to query the results with SQL.

Cost: 100 reviews per model with `gpt-4o-mini` and `gemini-2.0-flash` costs well under $0.10.
Raise `--llm-sample` once you've checked the output.

## Command-line options

| Flag | Meaning |
| --- | --- |
| `--input PATH` | Raw reviews CSV (required) |
| `--max-rows N` | Only read the first N rows (the Kaggle file has 568k) |
| `--local` | Skip S3 entirely |
| `--from-s3` | Download the input from S3 instead of uploading it |
| `--mock-llm` | Offline stand-in for OpenAI/Gemini, for testing |
| `--llm-sample N` | Reviews sent to each LLM (default 100) |

## Project layout

```
run_pipeline.py        entry point that runs all six steps
src/config.py          settings from .env
src/s3_io.py           S3 upload / download
src/clean.py           Pandas cleaning and labelling
src/llm_enrich.py      OpenAI + Gemini extraction and comparison
src/ml_models.py       TF-IDF + Logistic Regression / Decision Tree / XGBoost
src/forecast.py        monthly series + ARIMA back-test and forecast
src/dashboard.py       Plotly HTML dashboard
sql/athena_queries.sql Athena tables and example queries
data/make_sample.py    synthetic test data
```

## Results

Fill this in after your real run (numbers are in `outputs/summary.json`):

- Best ML model: ______ (F1 ____, ROC-AUC ____)
- LLM sentiment accuracy: OpenAI ____ vs Gemini ____
- ARIMA back-test MAE: ____ reviews/month
