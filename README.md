# Customer Sentiment Analysis & Voice-of-Customer Intelligence
**Portfolio Project 005** — NLP + Text Analytics + Early-Warning Issue Detection

## Business question
What are customers saying, why are they unhappy, which issues are getting
worse, and what should the business fix first?

## What this project demonstrates (new vs. Projects 001–004)

| Project | Business problem | Core new capability |
|---|---|---|
| 001 | E-Commerce Profitability | Business/data analytics |
| 002 | SaaS Churn & Retention | ML + causal/uplift |
| 003 | Retail Demand & Inventory | Forecasting + optimization |
| 004 | Financial Fraud & Risk | Anomaly detection + explainable AI |
| **005** | **Voice of Customer** | **NLP, text analytics, early-warning issue detection** |

## Scale note
The original brief sketches 1–2M interactions. This build uses **120,000
feedback interactions** across **8,000 customers**, **150 products**, and
**168,000 orders**, spanning **Jan 2024 – Jun 2026**. That's enough volume
to demonstrate every technique at realistic statistical scale (trend
detection, spike detection, topic recovery) while staying practical to
generate, process, and re-run end-to-end in minutes rather than hours.
The pipeline scales horizontally — swap in a larger `N_FEEDBACK` in
`scripts/02_generate_feedback_text.py` and everything downstream runs
unchanged.

## Repository structure
```
scripts/    01-09  — the full pipeline, run in order (see below)
data/              — generated datasets (customers, products, orders, feedback + all
                      enriched versions produced by each pipeline stage)
sql/               — analytics_queries.sql: 9 structured-analytics queries
powerbi_export/    — one flat CSV per dashboard page, ready to import into Power BI
outputs/           — model comparison results, evaluation metrics, alerts, explainability samples
docs/              — data dictionary, executive summary
```

## Pipeline (run in order)
```
python3 scripts/01_generate_data.py                       # customers, products, orders
python3 scripts/02_generate_feedback_text.py               # multi-channel feedback text + ground truth
python3 scripts/03_sentiment_classification.py              # VADER vs TF-IDF+LogReg vs Linear SVM
python3 scripts/04_aspect_sentiment.py                       # aspect-based sentiment (2 approaches compared)
python3 scripts/05_topic_discovery.py                        # NMF vs LDA vs embedding+KMeans
python3 scripts/06_emerging_issue_detection.py               # rolling-baseline spike detection
python3 scripts/07_priority_scoring_explainability.py        # priority engine + explainability cards
python3 scripts/09_powerbi_export.py                         # Power BI-ready flat tables
```
`scripts/08_validate_sql.py` runs all SQL queries against DuckDB as a
correctness check — not part of the main pipeline.

## 1. Sentiment analysis — results
| Model | Macro F1 | Precision | Recall |
|---|---|---|---|
| VADER (rule-based baseline) | 0.488 | 0.543 | 0.521 |
| TF-IDF + Logistic Regression | 1.000 | 1.000 | 1.000 |
| TF-IDF + Linear SVM | 0.9998 | 0.9999 | 0.9997 |

**Caveat, stated plainly:** the near-perfect ML scores are a ceiling effect
of template-based synthetic text — each sentiment class draws from a fixed,
non-overlapping phrase vocabulary, so TF-IDF trivially separates them. This
is expected and is *by design* for validating the pipeline mechanics; it is
not a claim that a production model would hit 100% macro F1 on real,
messy customer language (sarcasm, mixed signals, typos, and ambiguous
phrasing typically bring substantially below these synthetic ceiling scores for
3-class sentiment). VADER's weaker score is realistic and is the reason a
trained model is worth the investment.

### Aspect-based sentiment
Full text example from the spec: *"The product is excellent, but delivery
took two weeks and customer support never replied."* → Product Quality:
Positive, Delivery: Negative, Customer Support: Negative.

Two detection approaches were compared and evaluated against embedded
ground truth:
| Approach | Aspect-sentiment agreement |
|---|---|
| A: naive keyword spotting + VADER | 46.6% |
| B: curated phrase-lexicon matching | 99.98% |

This comparison is itself the finding: naive keyword+generic-sentiment
tagging is not reliable enough for business use. A curated aspect lexicon
(or, in production, a fine-tuned ABSA classifier) is what actually makes
aspect-level sentiment trustworthy.

## 2. Topic discovery — results
Purity against the 10 embedded ground-truth themes:
| Method | Purity |
|---|---|
| TF-IDF + NMF | **0.538** (best) |
| Embedding (SVD/LSA) + KMeans | 0.415 |
| LDA | 0.353 |

Purity below ~0.6 for all three methods is expected and informative: most
feedback messages in this dataset combine 1–3 themes in one message (as
real feedback does), so single-topic-per-document unsupervised models
can only ever partially disentangle them. The practical implication for
business use: **topic modeling is well-suited to macro-level theme
frequency and trend tracking (how much of everything is about delivery
this month), while per-message aspect tagging should rely on the phrase
lexicon / ABSA approach from Section 1** — not on forcing a single
dominant topic per message.

## 3. Emerging-issue detection
Rolling 30-day baseline + 7-day recent window, z-score based. On this run,
the operational detector (using lexicon-inferred themes from observable text, not synthetic ground-truth labels) correctly and exclusively caught the deliberately injected
spike, with no false positives on the other tracked themes:

> 🚨 **Emerging Issue Detected**: Payment Issues (Double Charge) complaints
> rose from a baseline of **0.71%** to **7.45%** of feedback over the last
> 7 days (+954%, z=9.9).

## 4. Customer experience impact
`sql/analytics_queries.sql` query 7 quantifies sentiment vs. churn-risk
association (using the customer's tier-based churn probability, nudged by
observed negative sentiment, as the outcome proxy — there is no separate
"did they actually churn" table in this synthetic dataset). As in prior
portfolio projects, this is reported as an **association**, not a causal
claim; the dataset wasn't designed with the identification strategy
(instruments, natural experiment, etc.) a causal claim would require.

## 5. Complaint severity & prioritization
Priority score (0–100) blends sentiment severity, theme severity, customer
value, churn risk, active-emerging-issue membership, and repeat-complaint
history. Distribution on this run:
- Critical: 516 | High: 49,075 | Medium: 58,664 | Low: 11,745
- 96.5% of Critical/High items are negative-sentiment feedback
  (sanity check against ever ranking positive feedback as urgent)

Example explainability card (`outputs/explainability_sample_top25.json`):
```json
{
  "sentiment": "Negative — model confidence 100%",
  "primary_issue": "Payment Issues (Double Charge)",
  "secondary_issues": ["Feature Requests", "Payment Issues"],
  "severity": "Critical",
  "customer_value": "Platinum",
  "repeat_complaints_before_this": 10,
  "part_of_emerging_issue": true,
  "recommended_priority": "Critical",
  "priority_score": 85.1
}
```

## 6. ML/NLP evaluation
Covered above; full metrics (per-class precision/recall, confusion
matrices) are in `outputs/sentiment_model_comparison.json`,
`outputs/topic_discovery_evaluation.json`, and
`outputs/aspect_sentiment_evaluation.json`.

## 7. Explainability
Every scored message carries a business-readable card (sentiment +
model confidence, primary/secondary issue inferred from observable text, severity, customer value, repeat
history, emerging-issue flag, recommended priority) — see
`outputs/explainability_sample_top25.json` for the top 25 by priority
score.

## 8. SQL analytics
`sql/analytics_queries.sql` — 9 validated queries (via DuckDB, see
`scripts/08_validate_sql.py`): sentiment trend by month, products with
worsening feedback, complaint rate by category/channel, repeat-complaint
customers, customer 360 history, channel performance, sentiment vs. churn
risk, SLA/resolution proxy, and a 45-day emerging-issue drill-down feed.

## 9. Power BI — Voice of Customer Command Center
`powerbi_export/` contains one pre-aggregated, import-ready CSV per page:
`page1_executive_overview`, `page2_sentiment_trends`,
`page3_topic_intelligence`, `page4_emerging_issues_*`,
`page5_product_experience`, `page6_support_experience`,
`page7_customer_risk`, `page8_action_priority_queue`, plus a shared
`fact_feedback.csv` for any cross-page relationships/DAX measures you want
to add on top.

## Known limitations (stated for transparency)
- Synthetic text is template-based, not LLM-generated free text — this
  keeps generation fast and gives exact ground truth, but caps realism
  and inflates supervised-model scores (see Section 1 caveat).
- No separate "actual churn" or "actual return" event table exists in this
  dataset, so churn/return linkage is a proxy/association, not validated
  against a real outcome.
- Topic-purity ground truth uses each message's *first-listed* embedded
  theme as "the" dominant theme; multi-theme messages are inherently
  ambiguous for any single-label evaluation.


## Publication integrity
Synthetic ground-truth fields (`true_themes`, `true_aspect_sentiments`, and `true_overall_sentiment`) are retained for offline evaluation only. Operational emerging-issue detection and complaint prioritisation use `inferred_themes` derived from observable feedback text. This prevents privileged synthetic labels from leaking into business-facing outputs.

The sentiment model's confidence is a model score on deliberately template-based synthetic text and is **not presented as a calibrated real-world probability**.
