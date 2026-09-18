"""
Aspect-Based Sentiment Analysis (ABSA) — two approaches compared:

  A. Naive keyword spotting + VADER sentence-level sentiment
     (fast, zero curation cost, but noisy — sentence-splitting and VADER's
     generic lexicon don't always agree with domain-specific phrasing)

  B. Curated phrase-lexicon matching
     (maps known domain phrases directly to an aspect + sentiment; higher
     accuracy but requires the lexicon to be built/maintained)

Both are evaluated against the embedded ground truth aspect/sentiment pairs,
which is only possible because this is synthetic data with known answers.
The comparison itself is the deliverable: it shows *why* a production ABSA
system needs curated aspect lexicons (or a trained classifier) rather than
generic keyword+sentiment heuristics.
"""
import pandas as pd
import re
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("data/feedback_with_sentiment.csv")
analyzer = SentimentIntensityAnalyzer()

THEME_TO_ASPECT = {
    "delivery_delay": "Delivery",
    "product_quality": "Product Quality",
    "customer_support": "Customer Support",
    "pricing": "Pricing",
    "refund_problems": "Refunds",
    "app_performance": "App/Website",
    "packaging": "Packaging",
    "payment_issue_general": "Payment",
    "payment_double_charge": "Payment",
    "feature_request": "Feature Requests",
    "general_satisfaction": None,
}
SENT_MAP = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}

ASPECT_KEYWORDS = {
    "Delivery": ["delivery", "shipping", "shipped", "package", "arrived", "transit", "courier"],
    "Product Quality": ["quality", "broke", "damaged", "flimsy", "durable", "material", "build"],
    "Customer Support": ["support", "agent", "customer service", "hold", "representative"],
    "Pricing": ["price", "pricing", "overcharged", "value", "discount", "expensive", "cost"],
    "Refunds": ["refund", "reimburse", "money back"],
    "App/Website": ["app", "website", "site", "login", "checkout page", "crash", "glitch", "freeze"],
    "Packaging": ["packaging", "box", "wrapped", "crushed"],
    "Payment": ["payment", "charged", "charge", "card", "transaction", "deducted", "currency"],
    "Feature Requests": ["wish", "would be great", "would love", "feature", "option to", "dark mode"],
}


def split_fragments(text):
    return [p for p in re.split(r'(?<=[.!])\s+|\s+(?=[Bb]ut\b|[Hh]owever\b)', text) if p.strip()]


def naive_detect(text):
    fragments = split_fragments(text)
    found = {}
    text_lower = text.lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                frag = next((f for f in fragments if kw in f.lower()), text)
                score = analyzer.polarity_scores(frag)["compound"]
                label = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
                found[aspect] = label
                break
    return found


with open("data/theme_phrase_lexicon.json") as f:
    lexicon = json.load(f)


def lexicon_detect(text):
    text_lower = text.lower()
    found = {}
    for phrase, theme_sent in lexicon.items():
        if phrase in text_lower:
            theme, sentiment = theme_sent.split(":")
            aspect = THEME_TO_ASPECT.get(theme)
            if aspect:
                found[aspect] = SENT_MAP[sentiment]
    return found

def infer_themes(text):
    """Infer operational theme labels from observable text only.

    Synthetic true_themes remains in the dataset strictly for evaluation; downstream
    alerts and prioritisation must not consume it.
    """
    text_lower = text.lower()
    themes = []
    for phrase, theme_sent in lexicon.items():
        if phrase in text_lower:
            theme, _ = theme_sent.split(":")
            if theme not in themes:
                themes.append(theme)
    return "|".join(themes) if themes else "unclassified"


def evaluate(detect_fn, label):
    correct, total = 0, 0
    all_rows = []
    for _, row in df.iterrows():
        detected = detect_fn(row["text"])
        true_pairs = row["true_aspect_sentiments"].split("|") if isinstance(row["true_aspect_sentiments"], str) else []
        for tp in true_pairs:
            theme, sentiment = tp.split(":")
            mapped_aspect = THEME_TO_ASPECT.get(theme)
            if mapped_aspect is None:
                continue
            total += 1
            if detected.get(mapped_aspect) == SENT_MAP[sentiment]:
                correct += 1
        all_rows.append(detected)
    acc = correct / total if total else 0
    print(f"{label}: aspect-sentiment agreement = {acc:.3f} ({correct}/{total})")
    return acc, all_rows


acc_naive, _ = evaluate(naive_detect, "Approach A (keyword+VADER)")
acc_lexicon, lexicon_rows = evaluate(lexicon_detect, "Approach B (phrase lexicon)")

detected_strs = ["|".join(f"{a}:{s}" for a, s in d.items()) if d else "None Detected" for d in lexicon_rows]
df["detected_aspects"] = detected_strs
df["n_aspects_detected"] = [len(d) for d in lexicon_rows]
df["inferred_themes"] = df["text"].apply(infer_themes)
df.to_csv("data/feedback_with_aspects.csv", index=False)

with open("outputs/aspect_sentiment_evaluation.json", "w") as f:
    json.dump({
        "approach_a_keyword_vader_agreement": acc_naive,
        "approach_b_phrase_lexicon_agreement": acc_lexicon,
        "recommended_approach": "B (phrase lexicon)",
        "production_note": "Real-world deployments would need either a continuously "
                            "curated aspect-phrase lexicon or a fine-tuned ABSA "
                            "classifier (e.g. spaCy + custom NER, or a transformer "
                            "ABSA model) since customer language varies far more than "
                            "this synthetic template set.",
    }, f, indent=2)

print("Saved data/feedback_with_aspects.csv and outputs/aspect_sentiment_evaluation.json")
