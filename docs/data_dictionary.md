# Data Dictionary

## customers.csv
| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| signup_date | Date the customer joined |
| region | One of 5 geographic regions |
| tier | Bronze / Silver / Gold / Platinum |
| lifetime_value | Simulated lifetime spend, correlated with tier |
| base_churn_prob | Ground-truth churn probability by tier (used to derive churn_risk_score) |

## products.csv
| Column | Description |
|---|---|
| product_id | Unique product identifier |
| category | One of 8 product categories |
| price | Unit price |
| quality_risk | normal / elevated / high — ground-truth quality-issue propensity |

## orders.csv
| Column | Description |
|---|---|
| order_id, customer_id, product_id, order_date, order_value | Standard order backbone linking customers, products, and feedback |

## feedback.csv (base) → feedback_final_scored.csv (fully enriched, final)
| Column | Description |
|---|---|
| feedback_id | Unique feedback identifier |
| customer_id, product_id | Foreign keys |
| channel | review / support_ticket / survey / chat |
| feedback_date | Date of the interaction |
| text | The generated feedback text |
| true_themes | **Ground truth** (pipe-separated) — themes embedded by the generator, used only for evaluation |
| true_aspect_sentiments | **Ground truth** theme:sentiment pairs |
| true_overall_sentiment | **Ground truth** overall sentiment |
| predicted_sentiment, sentiment_confidence | Output of the winning sentiment model (script 03) |
| detected_aspects, n_aspects_detected | Output of the phrase-lexicon ABSA approach (script 04) |
| discovered_topic_id, discovered_topic_label | Output of the winning topic model (script 05) |
| primary_theme, theme_label_primary, theme_severity | Derived for scoring |
| tier, lifetime_value, base_churn_prob | Joined from customers |
| repeat_negative_count | Running count of this customer's prior negative feedback |
| sentiment_severity_score, customer_value_score, churn_risk_score, issue_growth_score, repeat_complaint_score | Priority engine components (0-1 each) |
| priority_score, priority_tier | Final output (0-100, Low/Medium/High/Critical) |

## daily_theme_trends.csv
Daily share of feedback volume per theme — feeds the emerging-issue
detector and the Power BI trend page.

## powerbi_export/*.csv
See README section 9 — one file per dashboard page, pre-aggregated so no
DAX logic is required to reproduce the headline numbers (though DAX
measures can be layered on top of `fact_feedback.csv` for interactivity).
