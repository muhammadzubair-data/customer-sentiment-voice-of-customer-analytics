"""
Emerging Issue Detection

For each theme, tracks its daily share of total feedback volume, computes a
rolling 30-day baseline mean + std, and flags any 7-day window where the
theme's share has spiked well beyond its historical baseline (z-score based,
business-readable output). This is the early-warning system described in
the spec (payment complaints example).
"""
import pandas as pd
import numpy as np
import json

df = pd.read_csv("data/feedback_with_topics.csv")
df["feedback_date"] = pd.to_datetime(df["feedback_date"])

THEME_LABELS = {
    "delivery_delay": "Delivery Delay", "product_quality": "Product Quality",
    "customer_support": "Customer Service", "pricing": "Pricing",
    "refund_problems": "Refund Problems", "app_performance": "App Performance",
    "packaging": "Packaging", "payment_issue_general": "Payment Issues",
    "payment_double_charge": "Payment Issues (Double Charge)",
    "feature_request": "Feature Requests", "general_satisfaction": "General Satisfaction",
}

# Operational detector uses themes inferred from observable text only.
# true_themes is retained strictly for offline evaluation of the synthetic generator.
df["theme_list"] = df["inferred_themes"].str.split("|")
exploded = df.explode("theme_list")[["feedback_date", "theme_list"]]
exploded.columns = ["feedback_date", "theme"]

daily_total = df.groupby(df["feedback_date"].dt.date).size().rename("total_feedback")
daily_theme = exploded.groupby([exploded["feedback_date"].dt.date, "theme"]).size().rename("theme_count")
daily_theme = daily_theme.reset_index()
daily_theme = daily_theme.merge(daily_total.rename_axis("feedback_date").reset_index(), on="feedback_date")
daily_theme["theme_share"] = daily_theme["theme_count"] / daily_theme["total_feedback"]
daily_theme["feedback_date"] = pd.to_datetime(daily_theme["feedback_date"])
daily_theme = daily_theme.sort_values(["theme", "feedback_date"])

alerts = []
for theme, grp in daily_theme.groupby("theme"):
    grp = grp.set_index("feedback_date")[["theme_share"]].asfreq("D", fill_value=0)
    grp["theme_share"] = grp["theme_share"].fillna(0)
    # baseline: trailing 30-day window, EXCLUDING the most recent 7 days being evaluated
    grp["baseline_mean"] = grp["theme_share"].shift(7).rolling(30, min_periods=14).mean()
    grp["baseline_std"] = grp["theme_share"].shift(7).rolling(30, min_periods=14).std()
    grp["recent_7d_avg"] = grp["theme_share"].rolling(7).mean()
    grp["z_score"] = (grp["recent_7d_avg"] - grp["baseline_mean"]) / grp["baseline_std"].replace(0, np.nan)
    grp["pct_change_vs_baseline"] = (grp["recent_7d_avg"] - grp["baseline_mean"]) / grp["baseline_mean"].replace(0, np.nan)

    latest = grp.iloc[-1]
    if pd.notna(latest["z_score"]) and latest["z_score"] > 3 and latest["pct_change_vs_baseline"] > 1.0:
        alerts.append({
            "theme": THEME_LABELS.get(theme, theme),
            "baseline_share_pct": round(latest["baseline_mean"] * 100, 2),
            "recent_7d_share_pct": round(latest["recent_7d_avg"] * 100, 2),
            "pct_change": round(latest["pct_change_vs_baseline"] * 100, 1),
            "z_score": round(latest["z_score"], 2),
            "severity": "Critical" if latest["z_score"] > 6 else "High",
            "message": (f"\U0001F6A8 Emerging Issue Detected: {THEME_LABELS.get(theme, theme)} "
                        f"complaints rose from a baseline of {latest['baseline_mean']*100:.2f}% "
                        f"to {latest['recent_7d_avg']*100:.2f}% of feedback over the last 7 days "
                        f"(+{latest['pct_change_vs_baseline']*100:.0f}%, z={latest['z_score']:.1f})."),
        })

alerts = sorted(alerts, key=lambda a: -a["z_score"])
print(f"Detected {len(alerts)} emerging issue(s):\n")
for a in alerts:
    print(a["message"])

with open("outputs/emerging_issue_alerts.json", "w") as f:
    json.dump(alerts, f, indent=2)

# also save the full daily theme share time series for the Power BI layer
daily_theme["theme_label"] = daily_theme["theme"].map(THEME_LABELS)
daily_theme.to_csv("data/daily_theme_trends.csv", index=False)
print("\nSaved outputs/emerging_issue_alerts.json and data/daily_theme_trends.csv")
