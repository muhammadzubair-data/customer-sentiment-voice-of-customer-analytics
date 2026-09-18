"""
Generates the text feedback table: reviews, support tickets, survey comments,
and chat transcripts, each tagged with ground-truth aspect/theme + sentiment
labels (kept in a separate `ground_truth` table, not used by the "unsupervised"
models directly — only for evaluation).

Ground truth themes (aligned to spec):
  Delivery Delay, Refund Problems, Product Quality, Pricing,
  Customer Service, App Performance, Payment Issues, Packaging,
  Feature Requests, General Satisfaction (positive catch-all)

Emerging issue injection: the "Payment Issues -> double_charge" sub-theme
runs at a steady ~0.5% base rate for the whole series, then spikes to ~4.8%
of daily feedback volume in the final 7 days of the dataset — the exact
scenario described in the spec, for the emerging-issue detector to catch.
"""
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(7)
np.random.seed(7)

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 6, 30)
TOTAL_DAYS = (END_DATE - START_DATE).days
N_FEEDBACK = 120000
CHANNELS = ["review", "support_ticket", "survey", "chat"]
CHANNEL_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

customers = pd.read_csv("data/customers.csv")
products = pd.read_csv("data/products.csv")
orders = pd.read_csv("data/orders.csv")
orders["order_date"] = pd.to_datetime(orders["order_date"])

# ---------------------------------------------------------------------------
# Theme definitions: aspect name -> {sentiment: [phrase templates]}
# ---------------------------------------------------------------------------
THEMES = {
    "delivery_delay": {
        "negative": [
            "delivery took way longer than promised",
            "my package arrived over a week late",
            "shipping was delayed with no updates",
            "the order got stuck in transit for days",
            "delivery estimate was completely wrong",
        ],
        "neutral": [
            "delivery arrived close to the estimated date",
            "shipping took the usual amount of time",
        ],
        "positive": [
            "delivery was faster than expected",
            "the package arrived right on time",
            "shipping was super quick",
        ],
    },
    "product_quality": {
        "negative": [
            "the product quality feels cheap and flimsy",
            "it broke after just a few uses",
            "the item arrived damaged",
            "quality is noticeably worse than earlier orders",
            "materials feel low grade for the price",
        ],
        "neutral": [
            "the product is about what I expected",
            "quality is average, nothing special",
        ],
        "positive": [
            "the product quality is excellent",
            "build quality feels premium and durable",
            "really impressed with how well it's made",
        ],
    },
    "customer_support": {
        "negative": [
            "customer support never replied to my message",
            "support was rude and unhelpful",
            "I waited on hold for over an hour",
            "the agent couldn't resolve my issue at all",
            "support kept transferring me without solving anything",
        ],
        "neutral": [
            "support answered but it took a while",
            "customer service was okay, nothing memorable",
        ],
        "positive": [
            "customer support was fast and genuinely helpful",
            "the support agent resolved my issue in minutes",
            "really appreciated how patient the support team was",
        ],
    },
    "pricing": {
        "negative": [
            "the price feels way too high for what you get",
            "pricing keeps going up without any added value",
            "I feel overcharged compared to competitors",
        ],
        "neutral": [
            "pricing is fair, about what I'd expect",
        ],
        "positive": [
            "great value for the price",
            "pricing is very reasonable",
            "excellent discount made this a great deal",
        ],
    },
    "refund_problems": {
        "negative": [
            "refund still hasn't shown up after almost two weeks",
            "I was told the refund would be processed but nothing happened",
            "getting a refund has been an absolute nightmare",
            "refund request was denied for no clear reason",
        ],
        "neutral": [
            "refund process took the standard amount of time",
        ],
        "positive": [
            "refund was processed quickly and without hassle",
        ],
    },
    "app_performance": {
        "negative": [
            "the app keeps crashing every time I try to check out",
            "the website is painfully slow and glitchy",
            "I keep getting logged out randomly on the app",
            "the app freezes whenever I open my order history",
        ],
        "neutral": [
            "the app works fine most of the time",
        ],
        "positive": [
            "the app is smooth and easy to use",
            "loved how fast and intuitive the website was",
        ],
    },
    "packaging": {
        "negative": [
            "packaging was flimsy and the box was crushed",
            "the item wasn't protected well and got scratched in transit",
        ],
        "neutral": [
            "packaging was standard, nothing notable",
        ],
        "positive": [
            "packaging was sturdy and beautifully presented",
        ],
    },
    "feature_request": {
        "neutral": [
            "it would be great if you added a dark mode option",
            "wish there was a way to track multiple orders at once",
            "would love to see more size options for this item",
            "a saved-payment-method feature would really help",
        ],
    },
    "payment_issue_general": {
        "negative": [
            "my payment failed twice before finally going through",
            "I was charged in the wrong currency",
            "the checkout page kept rejecting my card for no reason",
        ],
        "neutral": [
            "payment processing took a bit longer than usual",
        ],
    },
    # This is the sub-theme we spike at the end of the series
    "payment_double_charge": {
        "negative": [
            "payment failed but the money was still deducted from my account",
            "I was charged twice for the same order and no refund yet",
            "the transaction showed as failed but my card was charged anyway",
            "money left my account even though the order shows as failed",
        ],
    },
    "general_satisfaction": {
        "positive": [
            "overall a really great experience, will definitely order again",
            "very happy with this purchase from start to finish",
            "everything about this order exceeded my expectations",
        ],
        "neutral": [
            "an okay experience overall, nothing stood out",
        ],
        "negative": [
            "overall this was a disappointing experience",
        ],
    },
}

THEME_NAMES = list(THEMES.keys())
# baseline mix (payment_double_charge deliberately tiny at baseline: ~0.5%)
BASE_THEME_WEIGHTS = {
    "delivery_delay": 0.16,
    "product_quality": 0.16,
    "customer_support": 0.14,
    "pricing": 0.08,
    "refund_problems": 0.09,
    "app_performance": 0.10,
    "packaging": 0.06,
    "feature_request": 0.05,
    "payment_issue_general": 0.08,
    "payment_double_charge": 0.005,
    "general_satisfaction": 0.075,
}
# renormalize
_total = sum(BASE_THEME_WEIGHTS.values())
BASE_THEME_WEIGHTS = {k: v / _total for k, v in BASE_THEME_WEIGHTS.items()}

CONNECTORS = [" Also, ", " On top of that, ", " In addition, ", " However, ", " But "]


def pick_sentiment_bucket(theme, forced_negative_bias=0.0):
    """Pick a sentiment for a given theme, respecting which sentiments exist for it."""
    available = list(THEMES[theme].keys())
    if "negative" in available and forced_negative_bias > 0:
        weights = []
        for s in available:
            if s == "negative":
                weights.append(0.55 + forced_negative_bias)
            elif s == "neutral":
                weights.append(0.25)
            else:
                weights.append(0.20)
        weights = np.array(weights) / sum(weights)
        return np.random.choice(available, p=weights)
    return random.choice(available)


def build_text(theme_sentiment_pairs):
    """Compose feedback text out of 1-3 (theme, sentiment) pairs."""
    phrases = []
    for theme, sentiment in theme_sentiment_pairs:
        phrase = random.choice(THEMES[theme][sentiment])
        phrases.append(phrase.capitalize() if not phrases else phrase)
    text = phrases[0]
    for p in phrases[1:]:
        text += random.choice(CONNECTORS).lower() if text.endswith(".") else random.choice(CONNECTORS)
        text += p
    if not text.endswith((".", "!")):
        text += "."
    return text[0].upper() + text[1:]


records = []
day_offsets = np.random.randint(0, TOTAL_DAYS, N_FEEDBACK)
day_offsets.sort()  # roughly chronological volume, we'll shuffle write order later

for i, day_off in enumerate(day_offsets):
    fb_date = START_DATE + timedelta(days=int(day_off))
    days_from_end = (END_DATE - fb_date).days

    # --- emerging issue injection window: last 7 days ---
    in_spike_window = days_from_end <= 7

    weights = dict(BASE_THEME_WEIGHTS)
    if in_spike_window:
        # push payment_double_charge from ~0.5% baseline to ~4.8%
        boost = 0.043
        weights["payment_double_charge"] += boost
        # take the boost proportionally from other themes
        other_keys = [k for k in weights if k != "payment_double_charge"]
        shrink = boost / len(other_keys)
        for k in other_keys:
            weights[k] = max(weights[k] - shrink, 0.001)
        s = sum(weights.values())
        weights = {k: v / s for k, v in weights.items()}

    n_themes = np.random.choice([1, 2, 3], p=[0.55, 0.35, 0.10])
    chosen_themes = list(np.random.choice(
        THEME_NAMES, size=n_themes, replace=False,
        p=[weights[t] for t in THEME_NAMES]
    ))

    pairs = []
    for t in chosen_themes:
        sentiment = pick_sentiment_bucket(t)
        pairs.append((t, sentiment))

    text = build_text(pairs)

    # overall sentiment = majority vote across aspect sentiments, ties -> negative-leaning
    sentiments = [s for _, s in pairs]
    if sentiments.count("negative") >= max(sentiments.count("positive"), sentiments.count("neutral")):
        overall = "negative" if "negative" in sentiments else ("neutral" if "neutral" in sentiments else "positive")
    elif sentiments.count("positive") > sentiments.count("neutral"):
        overall = "positive"
    else:
        overall = "neutral"

    channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
    cust_id = random.choice(customers["customer_id"].values)
    prod_id = random.choice(products["product_id"].values)

    records.append({
        "feedback_id": f"FB-{i+1:07d}",
        "customer_id": cust_id,
        "product_id": prod_id,
        "channel": channel,
        "feedback_date": fb_date,
        "text": text,
        "true_themes": "|".join(chosen_themes),
        "true_aspect_sentiments": "|".join(f"{t}:{s}" for t, s in pairs),
        "true_overall_sentiment": overall,
    })

feedback = pd.DataFrame(records)
feedback = feedback.sample(frac=1.0, random_state=11).reset_index(drop=True)  # shuffle row order
feedback["feedback_date"] = pd.to_datetime(feedback["feedback_date"])

print(feedback["true_overall_sentiment"].value_counts(normalize=True))
print("\nPayment double-charge theme rate, last 7 days vs baseline:")
feedback["is_spike_theme"] = feedback["true_themes"].str.contains("payment_double_charge")
last7_mask = feedback["feedback_date"] > (END_DATE - timedelta(days=7))
print("Last 7 days rate:", feedback.loc[last7_mask, "is_spike_theme"].mean())
print("Prior period rate:", feedback.loc[~last7_mask, "is_spike_theme"].mean())

feedback.drop(columns=["is_spike_theme"]).to_csv("data/feedback.csv", index=False)
print(f"\nSaved {len(feedback)} feedback records to data/feedback.csv")

# Export the phrase-level lexicon (phrase -> theme:sentiment) for downstream
# aspect-based sentiment evaluation scripts.
import json
lexicon = {}
for theme, sentiments in THEMES.items():
    for sentiment, phrases in sentiments.items():
        for phrase in phrases:
            lexicon[phrase.lower()] = f"{theme}:{sentiment}"
with open("data/theme_phrase_lexicon.json", "w") as f:
    json.dump(lexicon, f, indent=2)
print(f"Saved phrase lexicon with {len(lexicon)} entries to data/theme_phrase_lexicon.json")
