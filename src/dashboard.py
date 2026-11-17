"""Build a single self-contained HTML dashboard of the results with Plotly."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREY = "#2563EB", "#EA580C", "#94A3B8"


def build(ml_results: pd.DataFrame, history: pd.DataFrame, forecast: pd.DataFrame,
          llm_rows: pd.DataFrame, llm_stats: pd.DataFrame, out_path) -> None:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "ML models: F1 score on held-out reviews",
            "Monthly review volume with 6-month ARIMA forecast",
            "LLM sentiment accuracy (vs. star rating)",
            "Review topics found by the LLM",
        ),
        vertical_spacing=0.16, horizontal_spacing=0.1,
    )

    # 1. ML model comparison
    fig.add_bar(row=1, col=1, x=ml_results["model"], y=ml_results["f1"], marker_color=BLUE,
                text=ml_results["f1"], textposition="outside", showlegend=False)
    fig.update_yaxes(range=[0, 1.1], row=1, col=1)

    # 2. Forecast
    fig.add_scatter(row=1, col=2, x=history.index, y=history["reviews"], mode="lines",
                    name="Actual", line=dict(color=BLUE))
    fig.add_scatter(row=1, col=2, x=forecast.index, y=forecast["upper_80"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(row=1, col=2, x=forecast.index, y=forecast["lower_80"], mode="lines",
                    line=dict(width=0), fill="tonexty", fillcolor="rgba(234,88,12,0.15)",
                    name="80% interval")
    fig.add_scatter(row=1, col=2, x=forecast.index, y=forecast["forecast"], mode="lines",
                    name="Forecast", line=dict(color=ORANGE, dash="dash"))

    # 3. LLM comparison
    if not llm_stats.empty:
        fig.add_bar(row=2, col=1, x=llm_stats["model"], y=llm_stats["sentiment_accuracy"],
                    marker_color=[BLUE, ORANGE][: len(llm_stats)], showlegend=False,
                    text=llm_stats["sentiment_accuracy"], textposition="outside")
        fig.update_yaxes(range=[0, 1.1], row=2, col=1)

    # 4. Topics
    if not llm_rows.empty:
        topics = llm_rows["topic"].value_counts().sort_values()
        fig.add_bar(row=2, col=2, x=topics.values, y=topics.index, orientation="h",
                    marker_color=GREY, showlegend=False)

    fig.update_layout(
        title="Customer Review Insights",
        height=850, template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=0.55),
        margin=dict(t=110, l=60, r=30, b=50),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
