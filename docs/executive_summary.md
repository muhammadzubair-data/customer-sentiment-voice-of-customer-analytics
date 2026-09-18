# Executive Summary — Voice of Customer Intelligence (Project 005)

**Dataset:** 120,000 customer feedback interactions (reviews, support
tickets, surveys, chat) from 8,000 customers across 150 products, Jan 2024
– Jun 2026.

## Key findings
1. **42.7% of all feedback is negative**, concentrated in five themes:
   Product Quality, Delivery Delay, Customer Service, Payment Issues, and
   App Performance — together accounting for the large majority of
   negative volume.
2. **An emerging issue was caught in real time**: payment "double-charge"
   complaints (money deducted on a failed transaction) jumped from a
   steady 0.7% baseline to 7.5% of all feedback in the most recent 7-day
   window — a nearly 10x spike, flagged automatically before it could be
   missed in routine reporting.
3. **A curated aspect lexicon outperforms naive keyword tagging by more
   than 2x** for pinpointing which specific part of an order a customer is
   unhappy about (99.98% vs. 46.6% agreement with known intent) — the
   clearest lever for making sentiment analysis actionable rather than
   just descriptive.
4. **A trained classifier is worth the investment over rule-based
   sentiment scoring**: TF-IDF + Logistic Regression dramatically
   outperformed the VADER rule-based baseline (macro F1 essentially
   perfect on this data vs. 0.49 for VADER); real-world gains from
   training a model will be smaller but still substantial given VADER's
   weak baseline.
5. **The priority engine correctly isolates urgency**: 96% of everything
   ranked Critical/High priority is genuinely negative-sentiment feedback,
   and the single highest-priority case in the dataset is a Platinum-tier
   customer, on their 11th complaint, caught up in the active payment
   double-charge spike — exactly the kind of case a 500-ticket-per-day
   triage team needs surfaced first out of 20,000 daily messages.

## Recommended actions
- Route the payment double-charge spike to the payments team as an
  active incident — it is not noise; it is a statistically confirmed,
  fast-growing issue category.
- Adopt phrase-lexicon (or ML-based) aspect tagging over keyword spotting
  for any per-aspect reporting; keyword-only tagging is not reliable
  enough to act on.
- Use topic modeling for macro trend tracking (theme volume over time),
  and the aspect lexicon for per-message, per-aspect detail — the two
  techniques solve different problems and neither replaces the other.

## Caveats
Findings are drawn from synthetic data with embedded ground truth,
designed to validate that the analytics pipeline works correctly and
catches known patterns end-to-end. Absolute scores (e.g., near-perfect
sentiment classification) reflect the cleanliness of the synthetic text
and should not be read as expected real-world model performance — see the
README's "Known limitations" section for full detail. Sentiment-to-churn
and sentiment-to-return linkages are reported as associations, not causal
effects.
