"""Use OpenAI and Google Gemini to extract topic, sentiment and a summary from reviews,
then compare the two models on accuracy, speed and token usage."""
import json
import time

import pandas as pd

from . import config

TOPICS = ["taste", "price", "packaging", "delivery", "quality", "health", "other"]

PROMPT = """You analyse customer product reviews.
Return ONLY a JSON object with these keys:
  "topic": one of {topics}
  "sentiment": "positive" or "negative"
  "summary": a summary of the review in at most 12 words

Review:
\"\"\"{review}\"\"\""""


def _parse(raw: str) -> dict:
    """Parse the model's JSON reply, tolerating ```json fences."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"topic": "other", "sentiment": "unknown", "summary": ""}
    topic = data.get("topic", "other")
    return {
        "topic": topic if topic in TOPICS else "other",
        "sentiment": str(data.get("sentiment", "unknown")).lower(),
        "summary": data.get("summary", ""),
    }


# ---------- model callers: each returns (parsed_dict, tokens_used) ----------

def _call_openai(client, review: str):
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(topics=TOPICS, review=review)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    return _parse(resp.choices[0].message.content), tokens


def _call_gemini(client, review: str):
    from google.genai import types

    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=PROMPT.format(topics=TOPICS, review=review),
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    usage = getattr(resp, "usage_metadata", None)
    tokens = getattr(usage, "total_token_count", 0) or 0
    return _parse(resp.text), tokens


def _call_mock(_client, review: str):
    """Offline stand-in so the pipeline can be tested without API keys."""
    text = review.lower()
    neg_words = ["bad", "terrible", "awful", "stale", "disappoint", "broken", "worst", "never", "waste"]
    sentiment = "negative" if any(w in text for w in neg_words) else "positive"
    topic = next((t for t in TOPICS if t in text), "other")
    summary = " ".join(review.split()[:10])
    return {"topic": topic, "sentiment": sentiment, "summary": summary}, 0


def _make_clients(mock: bool):
    if mock:
        return {"openai": (None, _call_mock), "gemini": (None, _call_mock)}

    clients = {}
    if config.OPENAI_API_KEY:
        from openai import OpenAI

        clients["openai"] = (OpenAI(api_key=config.OPENAI_API_KEY), _call_openai)
    if config.GEMINI_API_KEY:
        from google import genai

        clients["gemini"] = (genai.Client(api_key=config.GEMINI_API_KEY), _call_gemini)
    if not clients:
        raise ValueError("Set OPENAI_API_KEY and/or GEMINI_API_KEY, or run with --mock-llm.")
    return clients


def enrich(df: pd.DataFrame, sample_size: int, mock: bool = False):
    """Run each available model on a sample of reviews.

    Returns (enriched_rows, comparison_table).
    """
    sample = df.sample(n=min(sample_size, len(df)), random_state=config.RANDOM_STATE)
    clients = _make_clients(mock)

    rows, stats = [], []
    for name, (client, call) in clients.items():
        print(f"  running {name} on {len(sample)} reviews...")
        correct = tokens_total = failures = 0
        start = time.perf_counter()
        for _, r in sample.iterrows():
            try:
                out, tokens = call(client, r["review_text"])
            except Exception as exc:  # keep going if one call fails
                failures += 1
                print(f"    {name} call failed for review {r['review_id']}: {exc}")
                continue
            tokens_total += tokens
            truth = "positive" if r["label"] == 1 else "negative"
            correct += out["sentiment"] == truth
            rows.append({"review_id": r["review_id"], "model": name, **out})
        elapsed = time.perf_counter() - start
        done = len(sample) - failures
        stats.append({
            "model": name,
            "reviews": done,
            "sentiment_accuracy": round(correct / done, 3) if done else 0.0,
            "avg_seconds_per_review": round(elapsed / max(done, 1), 3),
            "total_tokens": tokens_total,
            "failed_calls": failures,
        })

    return pd.DataFrame(rows), pd.DataFrame(stats)
