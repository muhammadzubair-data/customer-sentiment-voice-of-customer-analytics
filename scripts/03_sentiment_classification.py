"""
Sentiment classification model comparison:
  1. VADER (rule-based baseline, no training)
  2. TF-IDF + Logistic Regression
  3. TF-IDF + Linear SVM

Ground-truth label used only for evaluation: true_overall_sentiment.
Metrics: Macro F1, Precision, Recall, per-class report, confusion matrix.
"""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, f1_score, precision_score,
                              recall_score, confusion_matrix)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("data/feedback.csv")
LABELS = ["negative", "neutral", "positive"]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["true_overall_sentiment"], test_size=0.2,
    stratify=df["true_overall_sentiment"], random_state=42
)

results = {}

# --- 1. VADER baseline ---------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def vader_label(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

vader_preds = X_test.apply(vader_label)
results["vader_baseline"] = {
    "macro_f1": f1_score(y_test, vader_preds, average="macro"),
    "macro_precision": precision_score(y_test, vader_preds, average="macro"),
    "macro_recall": recall_score(y_test, vader_preds, average="macro"),
    "report": classification_report(y_test, vader_preds, labels=LABELS, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, vader_preds, labels=LABELS).tolist(),
}

# --- 2 & 3. TF-IDF + LogReg / Linear SVM ----------------------------------
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0)
logreg.fit(Xtr, y_train)
lr_preds = logreg.predict(Xte)
results["tfidf_logreg"] = {
    "macro_f1": f1_score(y_test, lr_preds, average="macro"),
    "macro_precision": precision_score(y_test, lr_preds, average="macro"),
    "macro_recall": recall_score(y_test, lr_preds, average="macro"),
    "report": classification_report(y_test, lr_preds, labels=LABELS, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, lr_preds, labels=LABELS).tolist(),
}

svm = LinearSVC(class_weight="balanced", C=1.0)
svm.fit(Xtr, y_train)
svm_preds = svm.predict(Xte)
results["tfidf_linear_svm"] = {
    "macro_f1": f1_score(y_test, svm_preds, average="macro"),
    "macro_precision": precision_score(y_test, svm_preds, average="macro"),
    "macro_recall": recall_score(y_test, svm_preds, average="macro"),
    "report": classification_report(y_test, svm_preds, labels=LABELS, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, svm_preds, labels=LABELS).tolist(),
}

print("Model comparison (macro F1):")
for name, r in results.items():
    print(f"  {name:20s} macro_f1={r['macro_f1']:.4f}  precision={r['macro_precision']:.4f}  recall={r['macro_recall']:.4f}")

with open("outputs/sentiment_model_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# --- Persist the best model's predictions over the FULL dataset for downstream use ---
best_name = max(results, key=lambda k: results[k]["macro_f1"])
print(f"\nBest model: {best_name}")

X_full = vectorizer.transform(df["text"])
if best_name == "tfidf_linear_svm":
    df["predicted_sentiment"] = svm.predict(X_full)
    # LinearSVC has no predict_proba; use decision_function margin as confidence proxy
    margins = svm.decision_function(X_full)
    conf = np.max(margins, axis=1) if margins.ndim > 1 else np.abs(margins)
    conf = (conf - conf.min()) / (conf.max() - conf.min())
    df["sentiment_confidence"] = np.round(0.5 + 0.5 * conf, 3)
else:
    model = logreg if best_name == "tfidf_logreg" else None
    if model is not None:
        df["predicted_sentiment"] = model.predict(X_full)
        proba = model.predict_proba(X_full)
        df["sentiment_confidence"] = np.round(proba.max(axis=1), 3)
    else:
        df["predicted_sentiment"] = df["text"].apply(vader_label)
        df["sentiment_confidence"] = 0.7

df.to_csv("data/feedback_with_sentiment.csv", index=False)
print("Saved data/feedback_with_sentiment.csv")
