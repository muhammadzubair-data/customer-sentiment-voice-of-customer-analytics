"""
Topic Discovery — three approaches compared:
  1. TF-IDF + NMF (Non-negative Matrix Factorization)
  2. LDA (Latent Dirichlet Allocation) on raw term counts
  3. Embedding-based clustering: TF-IDF -> TruncatedSVD (LSA embedding) -> KMeans
     (a lightweight embedding approach that runs fully offline; a
     transformer-embedding model would be a drop-in upgrade in production)

Because the synthetic generator embeds 10 known ground-truth themes, we can
score each method on how well its discovered clusters/topics recover those
themes (using purity against the dominant embedded theme per document) —
the "nice portfolio differentiator" called out in the spec.
"""
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans

df = pd.read_csv("data/feedback_with_aspects.csv")
texts = df["text"].tolist()

# dominant ground-truth theme per doc (first listed theme) for purity scoring
df["dominant_true_theme"] = df["true_themes"].apply(lambda s: s.split("|")[0])
N_TOPICS = 10

def purity_score(cluster_assignments, true_labels):
    result_df = pd.DataFrame({"cluster": cluster_assignments, "true": true_labels})
    total = len(result_df)
    correct = 0
    mapping = {}
    for c in result_df["cluster"].unique():
        sub = result_df[result_df["cluster"] == c]
        majority_label = sub["true"].mode()[0]
        mapping[c] = majority_label
        correct += (sub["true"] == majority_label).sum()
    return correct / total, mapping


def top_words(model, feature_names, n_top=8):
    topics = []
    for idx, comp in enumerate(model.components_):
        top_idx = comp.argsort()[-n_top:][::-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


results = {}

# --- 1. TF-IDF + NMF -------------------------------------------------------
tfidf_vec = TfidfVectorizer(max_features=5000, stop_words="english", min_df=5)
tfidf_matrix = tfidf_vec.fit_transform(texts)
nmf = NMF(n_components=N_TOPICS, random_state=42, init="nndsvda", max_iter=400)
nmf_topics = nmf.fit_transform(tfidf_matrix)
nmf_assignments = nmf_topics.argmax(axis=1)
purity, nmf_mapping = purity_score(nmf_assignments, df["dominant_true_theme"])
results["nmf"] = {
    "purity_vs_ground_truth_theme": purity,
    "top_words_per_topic": top_words(nmf, tfidf_vec.get_feature_names_out()),
}
print(f"NMF purity vs ground-truth theme: {purity:.3f}")

# --- 2. LDA -----------------------------------------------------------------
count_vec = CountVectorizer(max_features=5000, stop_words="english", min_df=5)
count_matrix = count_vec.fit_transform(texts)
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, max_iter=15,
                                 learning_method="online", batch_size=2048)
lda_topics = lda.fit_transform(count_matrix)
lda_assignments = lda_topics.argmax(axis=1)
purity_lda, lda_mapping = purity_score(lda_assignments, df["dominant_true_theme"])
results["lda"] = {
    "purity_vs_ground_truth_theme": purity_lda,
    "top_words_per_topic": top_words(lda, count_vec.get_feature_names_out()),
}
print(f"LDA purity vs ground-truth theme: {purity_lda:.3f}")

# --- 3. SVD embedding + KMeans ----------------------------------------------
svd = TruncatedSVD(n_components=100, random_state=42)
embedding = svd.fit_transform(tfidf_matrix)
kmeans = KMeans(n_clusters=N_TOPICS, random_state=42, n_init=10)
km_assignments = kmeans.fit_predict(embedding)
purity_km, km_mapping = purity_score(km_assignments, df["dominant_true_theme"])
results["embedding_kmeans"] = {
    "purity_vs_ground_truth_theme": purity_km,
    "method_detail": "TF-IDF -> TruncatedSVD(100-d LSA embedding) -> KMeans(k=10)",
}
print(f"Embedding+KMeans purity vs ground-truth theme: {purity_km:.3f}")

best_method = max(results, key=lambda k: results[k]["purity_vs_ground_truth_theme"])
print(f"\nBest topic discovery method: {best_method}")

# Map human-readable theme names for the business layer
THEME_LABELS = {
    "delivery_delay": "Delivery Delay", "product_quality": "Product Quality",
    "customer_support": "Customer Service", "pricing": "Pricing",
    "refund_problems": "Refund Problems", "app_performance": "App Performance",
    "packaging": "Packaging", "payment_issue_general": "Payment Issues",
    "payment_double_charge": "Payment Issues (Double Charge)",
    "feature_request": "Feature Requests", "general_satisfaction": "General Satisfaction",
}

# Persist the winning method's topic assignment onto the dataset, mapped
# to its majority ground-truth theme label (this is what a real deployment
# would hand-label once, based on top words per topic)
if best_method == "nmf":
    assignments, mapping = nmf_assignments, nmf_mapping
elif best_method == "lda":
    assignments, mapping = lda_assignments, lda_mapping
else:
    assignments, mapping = km_assignments, km_mapping

df["discovered_topic_id"] = assignments
df["discovered_topic_label"] = [THEME_LABELS.get(mapping[a], mapping[a]) for a in assignments]
df.to_csv("data/feedback_with_topics.csv", index=False)

with open("outputs/topic_discovery_evaluation.json", "w") as f:
    json.dump({"method_comparison": results, "best_method": best_method}, f, indent=2, default=str)

print("Saved data/feedback_with_topics.csv and outputs/topic_discovery_evaluation.json")
