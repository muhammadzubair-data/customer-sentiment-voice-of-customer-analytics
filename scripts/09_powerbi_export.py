"""
Power BI Export Layer — "Voice of Customer Command Center"
Produces one flat, pre-aggregated CSV per dashboard page so Power BI just
needs to import and relate these tables (star-schema friendly), rather than
re-deriving logic in DAX.

Pages: Executive Overview, Sentiment Trends, Topic Intelligence,
Emerging Issues, Product Experience, Support Experience, Customer Risk,
Action Priority.
"""
import pandas as pd
import json

df = pd.read_csv("data/feedback_final_scored.csv")
df["feedback_date"] = pd.to_datetime(df["feedback_date"])
customers = pd.read_csv("data/customers.csv")
products = pd.read_csv("data/products.csv")

OUT = "powerbi_export/"

# --- Fact table (main, page-agnostic) --------------------------------------
fact_cols = ["feedback_id", "customer_id", "product_id", "channel", "feedback_date",
             "predicted_sentiment", "sentiment_confidence", "theme_label_primary",
             "priority_score", "priority_tier", "churn_risk_score", "tier", "lifetime_value"]
df[fact_cols].to_csv(f"{OUT}fact_feedback.csv", index=False)

# --- Page 1: Executive Overview (single-row KPI card table) -----------------
last_date = df["feedback_date"].max()
last_30 = df[df["feedback_date"] > last_date - pd.Timedelta(days=30)]
with open("outputs/emerging_issue_alerts.json") as f:
    alerts = json.load(f)

kpis = pd.DataFrame([{
    "total_feedback": len(df),
    "negative_sentiment_pct": round((df["predicted_sentiment"] == "negative").mean() * 100, 1),
    "csat_proxy_pct": round((df["predicted_sentiment"] == "positive").mean() * 100, 1),
    "emerging_issues_count": len(alerts),
    "critical_complaints_count": int((df["priority_tier"] == "Critical").sum()),
    "resolution_rate_proxy_pct": round(100 - (df["priority_tier"].isin(["Critical", "High"]).mean() * 100), 1),
    "at_risk_customers_count": int(df.loc[df["churn_risk_score"] > 0.5, "customer_id"].nunique()),
    "top_complaint_topic": df.loc[df["predicted_sentiment"] == "negative", "theme_label_primary"].mode()[0],
    "last_30d_feedback_volume": len(last_30),
    "as_of_date": str(last_date.date()),
}])
kpis.to_csv(f"{OUT}page1_executive_overview.csv", index=False)

# --- Page 2: Sentiment Trends (monthly) -------------------------------------
sentiment_trends = (df.groupby([df["feedback_date"].dt.to_period("M").astype(str), "predicted_sentiment"])
                      .size().reset_index(name="count"))
sentiment_trends.columns = ["month", "sentiment", "count"]
sentiment_trends.to_csv(f"{OUT}page2_sentiment_trends.csv", index=False)

# --- Page 3: Topic Intelligence (theme volume + avg sentiment) -------------
topic_intel = df.groupby("theme_label_primary").agg(
    volume=("feedback_id", "count"),
    negative_rate=("predicted_sentiment", lambda s: (s == "negative").mean()),
    avg_priority_score=("priority_score", "mean"),
).reset_index().sort_values("volume", ascending=False)
topic_intel.to_csv(f"{OUT}page3_topic_intelligence.csv", index=False)

# --- Page 4: Emerging Issues -------------------------------------------------
daily_trends = pd.read_csv("data/daily_theme_trends.csv")
daily_trends.to_csv(f"{OUT}page4_emerging_issues_daily_trends.csv", index=False)
pd.DataFrame(alerts).to_csv(f"{OUT}page4_emerging_issues_alerts.csv", index=False)

# --- Page 5: Product Experience ---------------------------------------------
product_exp = df.merge(products[["product_id", "category", "quality_risk"]], on="product_id", how="left")
product_summary = product_exp.groupby(["product_id", "category", "quality_risk"]).agg(
    feedback_volume=("feedback_id", "count"),
    negative_rate=("predicted_sentiment", lambda s: (s == "negative").mean()),
    avg_priority_score=("priority_score", "mean"),
).reset_index().sort_values("negative_rate", ascending=False)
product_summary.to_csv(f"{OUT}page5_product_experience.csv", index=False)

# --- Page 6: Support Experience (channel-level) -----------------------------
support_exp = df.groupby("channel").agg(
    volume=("feedback_id", "count"),
    negative_rate=("predicted_sentiment", lambda s: (s == "negative").mean()),
    avg_priority_score=("priority_score", "mean"),
    critical_count=("priority_tier", lambda s: (s == "Critical").sum()),
).reset_index()
support_exp.to_csv(f"{OUT}page6_support_experience.csv", index=False)

# --- Page 7: Customer Risk ----------------------------------------------------
customer_risk = df.groupby(["customer_id", "tier", "lifetime_value"]).agg(
    feedback_count=("feedback_id", "count"),
    negative_count=("predicted_sentiment", lambda s: (s == "negative").sum()),
    avg_churn_risk_score=("churn_risk_score", "mean"),
    max_priority_score=("priority_score", "max"),
).reset_index().sort_values("avg_churn_risk_score", ascending=False)
customer_risk.to_csv(f"{OUT}page7_customer_risk.csv", index=False)

# --- Page 8: Action Priority (queue for CX team) ----------------------------
action_queue = df[df["priority_tier"].isin(["Critical", "High"])].sort_values(
    "priority_score", ascending=False
)[["feedback_id", "customer_id", "tier", "channel", "feedback_date", "theme_label_primary",
   "predicted_sentiment", "priority_score", "priority_tier"]]
action_queue.to_csv(f"{OUT}page8_action_priority_queue.csv", index=False)

print("Power BI export layer complete. Files written to powerbi_export/:")
import os
for f in sorted(os.listdir(OUT)):
    print(" -", f)
