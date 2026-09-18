-- =============================================================================
-- Project 005 — Voice of Customer: SQL Analytics Layer
-- Tables: customers, products, orders, feedback_final_scored
-- Dialect: ANSI SQL / DuckDB-compatible (also runs on Postgres with trivial
-- changes to date-trunc syntax).
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Sentiment trend by month
-- -----------------------------------------------------------------------------
SELECT
    date_trunc('month', feedback_date) AS month,
    predicted_sentiment,
    COUNT(*) AS feedback_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY date_trunc('month', feedback_date)), 2) AS pct_of_month
FROM feedback_final_scored
GROUP BY 1, 2
ORDER BY 1, 2;

-- -----------------------------------------------------------------------------
-- 2. Products with worsening feedback (negative share trending up, last 90 vs prior 90 days)
-- -----------------------------------------------------------------------------
WITH recent AS (
    SELECT product_id,
           AVG(CASE WHEN predicted_sentiment = 'negative' THEN 1.0 ELSE 0 END) AS neg_rate_recent,
           COUNT(*) AS n_recent
    FROM feedback_final_scored
    WHERE feedback_date >= (SELECT MAX(feedback_date) FROM feedback_final_scored) - INTERVAL '90 days'
    GROUP BY product_id
),
prior AS (
    SELECT product_id,
           AVG(CASE WHEN predicted_sentiment = 'negative' THEN 1.0 ELSE 0 END) AS neg_rate_prior,
           COUNT(*) AS n_prior
    FROM feedback_final_scored
    WHERE feedback_date < (SELECT MAX(feedback_date) FROM feedback_final_scored) - INTERVAL '90 days'
      AND feedback_date >= (SELECT MAX(feedback_date) FROM feedback_final_scored) - INTERVAL '180 days'
    GROUP BY product_id
)
SELECT r.product_id, pr.category, r.n_recent, r.neg_rate_recent, pr2.neg_rate_prior,
       ROUND(r.neg_rate_recent - pr2.neg_rate_prior, 3) AS pct_point_change
FROM recent r
JOIN prior pr2 ON r.product_id = pr2.product_id
JOIN products pr ON pr.product_id = r.product_id
WHERE r.n_recent >= 20 AND pr2.n_prior >= 20
ORDER BY pct_point_change DESC
LIMIT 25;

-- -----------------------------------------------------------------------------
-- 3. Complaint rate by category (theme) and channel
-- -----------------------------------------------------------------------------
SELECT
    theme_label_primary AS issue_category,
    channel,
    COUNT(*) AS complaint_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_all_complaints
FROM feedback_final_scored
WHERE predicted_sentiment = 'negative'
GROUP BY 1, 2
ORDER BY complaint_count DESC;

-- -----------------------------------------------------------------------------
-- 4. Repeat complaints — customers with 3+ negative feedback events
-- -----------------------------------------------------------------------------
SELECT
    c.customer_id, c.tier, c.lifetime_value,
    COUNT(*) AS negative_feedback_count,
    MIN(f.feedback_date) AS first_complaint,
    MAX(f.feedback_date) AS latest_complaint
FROM feedback_final_scored f
JOIN customers c ON c.customer_id = f.customer_id
WHERE f.predicted_sentiment = 'negative'
GROUP BY c.customer_id, c.tier, c.lifetime_value
HAVING COUNT(*) >= 3
ORDER BY negative_feedback_count DESC, c.lifetime_value DESC
LIMIT 100;

-- -----------------------------------------------------------------------------
-- 5. Customer-level issue history (360 view for a given customer)
-- -----------------------------------------------------------------------------
SELECT
    f.feedback_id, f.feedback_date, f.channel, f.theme_label_primary,
    f.predicted_sentiment, f.priority_tier, f.priority_score
FROM feedback_final_scored f
WHERE f.customer_id = :customer_id   -- parameterize in application layer
ORDER BY f.feedback_date DESC;

-- -----------------------------------------------------------------------------
-- 6. Channel performance — average priority score and negative rate by channel
-- -----------------------------------------------------------------------------
SELECT
    channel,
    COUNT(*) AS total_feedback,
    ROUND(AVG(CASE WHEN predicted_sentiment = 'negative' THEN 1.0 ELSE 0 END), 3) AS negative_rate,
    ROUND(AVG(priority_score), 1) AS avg_priority_score,
    SUM(CASE WHEN priority_tier = 'Critical' THEN 1 ELSE 0 END) AS critical_count
FROM feedback_final_scored
GROUP BY channel
ORDER BY avg_priority_score DESC;

-- -----------------------------------------------------------------------------
-- 7. Sentiment vs. returns/churn proxy (association, not causal claim)
--    NOTE: this dataset does not include a separate returns table; the
--    association below is illustrative using churn_risk_score as the proxy
--    outcome, and should be read as "customers with X are more likely to
--    show elevated churn risk" — not a causal statement.
-- -----------------------------------------------------------------------------
SELECT
    predicted_sentiment,
    ROUND(AVG(churn_risk_score), 3) AS avg_churn_risk_score,
    COUNT(*) AS n
FROM feedback_final_scored
GROUP BY predicted_sentiment
ORDER BY avg_churn_risk_score DESC;

-- -----------------------------------------------------------------------------
-- 8. SLA / resolution-style performance proxy — median days between repeat
--    complaints on the same theme for the same customer (proxy for unresolved issues)
-- -----------------------------------------------------------------------------
WITH ordered AS (
    SELECT customer_id, theme_label_primary, feedback_date,
           LAG(feedback_date) OVER (PARTITION BY customer_id, theme_label_primary ORDER BY feedback_date) AS prev_date
    FROM feedback_final_scored
    WHERE predicted_sentiment = 'negative'
)
SELECT
    theme_label_primary,
    COUNT(*) AS repeat_events,
    ROUND(AVG(DATE_DIFF('day', prev_date, feedback_date)), 1) AS avg_days_between_repeat_complaints
FROM ordered
WHERE prev_date IS NOT NULL
GROUP BY theme_label_primary
ORDER BY repeat_events DESC;

-- -----------------------------------------------------------------------------
-- 9. Emerging issue snapshot — daily theme share, last 45 days (for BI drill-down)
-- -----------------------------------------------------------------------------
SELECT feedback_date, theme, theme_label, theme_share, theme_count, total_feedback
FROM daily_theme_trends
WHERE feedback_date >= (SELECT MAX(feedback_date) FROM daily_theme_trends) - INTERVAL '45 days'
ORDER BY theme, feedback_date;
