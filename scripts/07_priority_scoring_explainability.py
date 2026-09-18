"""
Complaint Severity & Prioritization + Explainability

Priority score (0-100) blends:
  - Sentiment severity   (how negative, weighted by model confidence)
  - Topic/theme severity (some themes are inherently higher-stakes)
  - Customer value       (lifetime value / tier)
  - Churn risk           (base churn probability by tier, nudged by sentiment)
  - Issue growth         (is this theme part of an active emerging-issue alert?)
  - Repeat complaints    (has this customer complained before?)

Output includes a business-readable explainability card per message.
"""
import pandas as pd
import numpy as np
import json

df = pd.read_csv("data/feedback_with_topics.csv")
customers = pd.read_csv("data/customers.csv")
df["feedback_date"] = pd.to_datetime(df["feedback_date"])

with open("outputs/emerging_issue_alerts.json") as f:
    alerts = json.load(f)
emerging_theme_labels = {a["theme"] for a in alerts}

THEME_SEVERITY = {
    "delivery_delay": 0.45, "product_quality": 0.65, "customer_support": 0.55,
    "pricing": 0.35, "refund_problems": 0.75, "app_performance": 0.50,
    "packaging": 0.30, "payment_issue_general": 0.70,
    "payment_double_charge": 0.95, "feature_request": 0.10,
    "general_satisfaction": 0.05,
}
THEME_LABELS = {
    "delivery_delay": "Delivery Delay", "product_quality": "Product Quality",
    "customer_support": "Customer Service", "pricing": "Pricing",
    "refund_problems": "Refund Problems", "app_performance": "App Performance",
    "packaging": "Packaging", "payment_issue_general": "Payment Issues",
    "payment_double_charge": "Payment Issues (Double Charge)",
    "feature_request": "Feature Requests", "general_satisfaction": "General Satisfaction",
}

df = df.merge(customers[["customer_id", "lifetime_value", "tier", "base_churn_prob"]],
              on="customer_id", how="left")

# repeat complaint count per customer (negative feedback only, prior to each message)
df = df.sort_values("feedback_date")
df["is_negative"] = (df["predicted_sentiment"] == "negative").astype(int)
df["repeat_negative_count"] = df.groupby("customer_id")["is_negative"].cumsum() - df["is_negative"]

def primary_theme(themes_str):
    return themes_str.split("|")[0]

df["primary_theme"] = df["inferred_themes"].apply(primary_theme)
df["theme_severity"] = df["primary_theme"].map(THEME_SEVERITY).fillna(0.4)
df["theme_label_primary"] = df["primary_theme"].map(THEME_LABELS)

# --- component scores (0-1 each) -------------------------------------------
sentiment_map = {"negative": 1.0, "neutral": 0.35, "positive": 0.0}
df["sentiment_severity_score"] = df["predicted_sentiment"].map(sentiment_map) * df["sentiment_confidence"]

df["customer_value_score"] = (df["lifetime_value"].rank(pct=True))
df["churn_risk_score"] = df["base_churn_prob"] * (1 + df["is_negative"] * 0.5)
df["churn_risk_score"] = df["churn_risk_score"].clip(upper=1.0)

df["issue_growth_score"] = df["theme_label_primary"].isin(emerging_theme_labels).astype(float)
df["repeat_complaint_score"] = np.tanh(df["repeat_negative_count"] / 3.0)  # saturates

# --- weighted priority score --------------------------------------------
WEIGHTS = {
    "sentiment_severity_score": 0.25,
    "theme_severity": 0.20,
    "customer_value_score": 0.15,
    "churn_risk_score": 0.15,
    "issue_growth_score": 0.15,
    "repeat_complaint_score": 0.10,
}
df["priority_score"] = sum(df[col] * w for col, w in WEIGHTS.items()) * 100
df["priority_score"] = df["priority_score"].round(1)

def priority_tier(score):
    if score >= 65:
        return "Critical"
    elif score >= 45:
        return "High"
    elif score >= 25:
        return "Medium"
    return "Low"

df["priority_tier"] = df["priority_score"].apply(priority_tier)

print("Priority tier distribution:")
print(df["priority_tier"].value_counts())
print(f"\nOnly negative/neutral feedback should dominate high priority — sanity check:")
print(df[df["priority_tier"].isin(["Critical", "High"])]["predicted_sentiment"].value_counts(normalize=True))

# --- explainability card (sample of highest-priority messages) -------------
def build_explainability_card(row):
    return {
        "feedback_id": row["feedback_id"],
        "sentiment": f"{row['predicted_sentiment'].capitalize()} — {row['sentiment_confidence']*100:.0f}%",
        "primary_issue": row["theme_label_primary"],
        "secondary_issues": [THEME_LABELS.get(t, t) for t in row["inferred_themes"].split("|")[1:] if t != "unclassified"],
        "severity": priority_tier(row["priority_score"]),
        "customer_value": row["tier"],
        "repeat_complaints_before_this": int(row["repeat_negative_count"]),
        "part_of_emerging_issue": row["theme_label_primary"] in emerging_theme_labels,
        "recommended_priority": row["priority_tier"],
        "priority_score": row["priority_score"],
    }

top_priority_sample = df.sort_values("priority_score", ascending=False).head(25)
explainability_cards = [build_explainability_card(r) for _, r in top_priority_sample.iterrows()]
with open("outputs/explainability_sample_top25.json", "w") as f:
    json.dump(explainability_cards, f, indent=2)

df.drop(columns=["is_negative"]).to_csv("data/feedback_final_scored.csv", index=False)
print("\nSaved data/feedback_final_scored.csv and outputs/explainability_sample_top25.json")
print("\nExample explainability card:")
print(json.dumps(explainability_cards[0], indent=2))
