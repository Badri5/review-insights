-- Query the pipeline's output in Amazon Athena.
-- Replace YOUR_BUCKET with your S3 bucket name, then run each statement in the Athena console.

CREATE DATABASE IF NOT EXISTS review_insights;

CREATE EXTERNAL TABLE IF NOT EXISTS review_insights.clean_reviews (
    review_id   BIGINT,
    `date`      STRING,
    rating      INT,
    review_text STRING,
    label       INT
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"')
LOCATION 's3://YOUR_BUCKET/processed/clean_reviews/'
TBLPROPERTIES ('skip.header.line.count' = '1');

CREATE EXTERNAL TABLE IF NOT EXISTS review_insights.llm_enriched (
    review_id BIGINT,
    model     STRING,
    topic     STRING,
    sentiment STRING,
    summary   STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"')
LOCATION 's3://YOUR_BUCKET/processed/llm_enriched/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- Monthly volume and average rating
SELECT substr(`date`, 1, 7)           AS month,
       count(*)                        AS reviews,
       round(avg(CAST(rating AS DOUBLE)), 2) AS avg_rating
FROM review_insights.clean_reviews
GROUP BY 1
ORDER BY 1;

-- Which topics drive negative reviews, per LLM
SELECT e.model, e.topic, count(*) AS negative_reviews
FROM review_insights.llm_enriched e
JOIN review_insights.clean_reviews c ON CAST(c.review_id AS BIGINT) = e.review_id
WHERE c.label = 0
GROUP BY e.model, e.topic
ORDER BY e.model, negative_reviews DESC;

-- How often OpenAI and Gemini agree on sentiment
SELECT count(*) AS reviews,
       sum(CASE WHEN o.sentiment = g.sentiment THEN 1 ELSE 0 END) AS agree
FROM review_insights.llm_enriched o
JOIN review_insights.llm_enriched g
  ON o.review_id = g.review_id AND o.model = 'openai' AND g.model = 'gemini';
