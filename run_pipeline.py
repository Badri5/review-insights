"""Customer Review Insights Pipeline: S3 -> Pandas -> GenAI -> ML -> forecast -> dashboard.

Examples
--------
Fully local, no API keys (good for a first test):
    python run_pipeline.py --input data/sample_reviews.csv --local --mock-llm

Real run with AWS S3, OpenAI and Gemini:
    python run_pipeline.py --input data/Reviews.csv --max-rows 50000
"""
import argparse
import json
from pathlib import Path

from src import clean, config, dashboard, forecast, llm_enrich, ml_models


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to the raw reviews CSV")
    p.add_argument("--max-rows", type=int, default=None, help="Only read the first N rows")
    p.add_argument("--local", action="store_true", help="Skip all S3 uploads/downloads")
    p.add_argument("--from-s3", action="store_true", help="Download the input from S3 instead of uploading it")
    p.add_argument("--mock-llm", action="store_true", help="Use an offline stand-in instead of OpenAI/Gemini")
    p.add_argument("--llm-sample", type=int, default=config.LLM_SAMPLE_SIZE,
                   help="How many reviews to send to each LLM")
    args = p.parse_args()

    out = config.OUTPUT_DIR
    raw_path = Path(args.input)
    raw_key = config.S3_RAW_PREFIX + raw_path.name

    # 1. Ingest
    print("[1/6] Ingest")
    if not args.local:
        from src import s3_io
        if args.from_s3:
            s3_io.download_file(raw_key, raw_path)
        else:
            s3_io.upload_file(raw_path, raw_key)

    # 2. Clean
    print("[2/6] Clean with Pandas")
    df = clean.load_and_clean(raw_path, max_rows=args.max_rows)
    df.to_csv(out / "clean_reviews.csv", index=False)
    print(f"  {len(df):,} reviews, {df['label'].mean():.0%} positive, "
          f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")

    # 3. GenAI enrichment
    print("[3/6] GenAI enrichment (OpenAI vs Gemini)")
    llm_rows, llm_stats = llm_enrich.enrich(df, args.llm_sample, mock=args.mock_llm)
    llm_rows.to_csv(out / "llm_enriched.csv", index=False)
    llm_stats.to_csv(out / "llm_comparison.csv", index=False)
    print(llm_stats.to_string(index=False))

    # 4. Classical ML
    print("[4/6] Train ML models")
    ml_results, top_words = ml_models.train_and_compare(df)
    ml_results.to_csv(out / "ml_results.csv", index=False)
    top_words.to_csv(out / "top_words.csv", index=False)
    print(ml_results.to_string(index=False))

    # 5. Time series forecast
    print("[5/6] ARIMA forecast")
    history, fc, mae = forecast.forecast_volume(df, config.FORECAST_HORIZON)
    history.to_csv(out / "monthly_history.csv")
    fc.to_csv(out / "forecast.csv")
    print(f"  back-test MAE: {mae} reviews/month")
    print(fc.to_string())

    # 6. Dashboard + upload results
    print("[6/6] Dashboard")
    dashboard.build(ml_results, history, fc, llm_rows, llm_stats, out / "dashboard.html")

    summary = {
        "reviews": len(df),
        "positive_share": round(float(df["label"].mean()), 3),
        "best_ml_model": ml_results.iloc[0].to_dict(),
        "forecast_backtest_mae": mae,
        "llm_comparison": llm_stats.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if not args.local:
        from src import s3_io
        for f in ["clean_reviews.csv", "llm_enriched.csv", "ml_results.csv", "forecast.csv", "summary.json"]:
            folder = f.rsplit(".", 1)[0]
            s3_io.upload_file(out / f, f"{config.S3_PROCESSED_PREFIX}{folder}/{f}")

    print(f"\nDone. Open {out / 'dashboard.html'} in a browser.")


if __name__ == "__main__":
    main()
